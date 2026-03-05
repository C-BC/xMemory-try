import json
import argparse
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Add project root to Python path
PROJECT_ROOT = os.path.expanduser("~/playground/AgentMem/clonemem_adaptation")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute llm-judge metrics (Choice Acc & LLM Judge)")
    parser.add_argument('--qa_file', type=str, 
                        default=f'data/new_data/all_users_benchmark_en.json',
                        help='Input file containing QA items and ground truth')
    parser.add_argument('--gen_file', type=str,
                        default=f'checkpoints/new_data/flatten/generation_results/周瑜_generation_top5_default_9fdf4fb7-c6d8-4fb1-9e01-581d5b85d84a_contriever-no_expansion-none-use_chunk.json',
                        help='Input file containing retrieval and generation results (JSON format)')
    parser.add_argument('--out_file', type=str, 
                        default=f'',
                        help='Output file to save detailed metrics')
    
    # LLM Judge args
    parser.add_argument('--model_name', default='gpt-4o', help="Model name for judge.")
    parser.add_argument('--base_url', default=None, help="Base URL for the API.")
    parser.add_argument('--api_key', default=None, help="API Key for the client.")
    parser.add_argument('--max_workers', type=int, default=16, help="Concurrency for LLM judge.")
    parser.add_argument('--eval_qa', action='store_true', help="Run LLM judge for QA.")
    parser.add_argument('--eval_mem', type=str, choices=['mem', 'media', 'none'], default='mem', help="Run LLM judge for Mem (mem, media, or none).")
    parser.add_argument('--topk', type=int, default=10, help="Top K items for mem evaluation (if eval_mem=mem).")

    return parser.parse_args()

QA_eval_prompt = """Your task is to evaluate the consistency between the [candidate_answer] and the [groundtruth_memory].

IMPORTANT:
- Focus only on whether “facts, constraints, preferences, and confirmed states” are correctly used
- Do NOT evaluate language style, tone, politeness, empathy, or fluency
- Do NOT give a high score just because the answer “sounds reasonable”
- The reference answer is only to help understand how relevant memory should ideally be used; a candidate answer does not need to exactly match the reference answer to receive a full score

---

### Evaluation Dimensions
QA_consistency_score:  
Scoring (0–3):
Score 0: The candidate answer contradicts the ground-truth memory, or contains factual content not covered or supported by the ground-truth memory.
Score 1: The candidate answer does not contradict the ground-truth memory, but is generic and does not rely on user memory.  
Score 2: The candidate answer uses part of the ground-truth memory as support.  
Score 3: The candidate answer uses all relevant user memory.

---

### Output Format
Please provide your evaluation results using the following structure:

```json
{{
  "QA_consistency_score": int,
  "QA_consistency_reason": str  # Explain the reason for the score and list content in the candidate answer that is based on or contradicts the groundtruth_memory
}}
```

### Input Data
• <question>: {question}
• <groundtruth_memory>: {groundtruth_memory}
• <reference_answer>: {reference_answer}
• <candidate_answer>: {candidate_answer}
"""

Mem_eval_prompt = """Your task is to evaluate the consistency between the [retrieved memory] and the [ground-truth memory], and whether the retrieved memory is helpful.

### Evaluation Dimensions
#### 1. Memory Recall
Mem_recall: Semantics-aware memory recall calculation (0–1)  
step1: For each groundtruth_memory, check in sequence whether its semantics are contained in any retrieved_memory.  
step2: Count how many groundtruth_memory items are covered (hits_cnt).  
step3: Compute the final recall score as hits_cnt / total number of groundtruth_memory items.

#### 2. Memory Helpfulness
Mem_helpful_score: The helpfulness of the retrieved memory for answering the question  
Score 0: retrieved_memory contains mutually conflicting or contradictory memories, which not only fail to help answer the question but may also cause confusion.  
Score 1: retrieved_memory is somewhat helpful for answering the question (can provide partial supporting evidence).  
Score 2: retrieved_memory is very helpful for answering the question (can provide comprehensive supporting evidence).

---

### Output Format
Please provide your evaluation results using the following structure:

```json
{{
  "Mem_recall": float,
  "Mem_helpful_score": int,
  "Mem_hits": list[str],  # List the matched groundtruth_memory items
  "Mem_helpful_reason": str  # Explain the reason for the score and list the retrieved_memory that helps answer the question
}}
```

---

### Input Data
• <question>: {question}
• <groundtruth_memory>: {groundtruth_memory}
• <retrieved_memory>: {retrieved_memory}
"""

def load_ground_truth(data):
    """
    Load ground truth from QA file.
    Returns: dict { qa_id: qa_item_dict }
    """
    qa_map = {}
    qa_items = data.get('qa_items', [])
    for item in qa_items:
        qa_map[item['id']] = item
    return qa_map

def extract_json_from_text(text):
    try:
        # Try finding JSON block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try finding brace enclosed JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except:
        return None

def run_llm_judge(client, model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return extract_json_from_text(content)
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return None

def evaluate_single_item(item, gt_item, client, model, eval_qa, eval_mem_type, topk):
    results = {}
    
    question = item.get('question', '')
    generated_answer = item.get('generated_answer', '')
    evidence_used = item.get('evidence_used', '')
    gold_answer = gt_item.get('answer', '')
    
    # Construct groundtruth memory string from gt_item['evidence']
    gt_evidence_list = gt_item.get('evidence', [])
    gt_mem_str = "\n".join([f"- {e.get('statement', '')}" for e in gt_evidence_list])
    
    # if eval_qa:
    qa_prompt = QA_eval_prompt.format(
        question=question,
        groundtruth_memory=gt_mem_str,
        reference_answer=gold_answer,
        candidate_answer=generated_answer
    )
    qa_result = run_llm_judge(client, model, qa_prompt)
    if qa_result:
        results.update(qa_result)
            
    if eval_mem_type != 'none':
        retrieved_memory = ""
        if eval_mem_type == 'media':
             # Assert no subquestions when using media mode
             # Media mode expects evidence_used to be formatted evidence text,
             # but with subquestions it contains the full prompt structure
             ranked_items = item.get('ranked_items', [])
             has_subquestions = any('subquestion' in item for item in ranked_items)
             if has_subquestions:
                 raise ValueError(
                     f"Cannot use eval_mem_type='media' with subquestion decomposition. "
                     f"Use eval_mem_type='mem' instead or disable subquestions in retrieval."
                 )
             retrieved_memory = item.get('evidence_used', '')
        elif eval_mem_type == 'mem':
             ranked_items = item.get('ranked_items', [])
             # Filter res_type != 'chunk'
             filtered = [x for x in ranked_items if x.get('res_type') != 'chunk']
             
             # Deduplicate by content (keep first occurrence)
             # Important when using subquestions where same memory may appear
             # multiple times from different subquestions
             seen_contents = set()
             deduplicated = []
             for x in filtered:
                 content = x.get('content', '').strip()
                 if content and content not in seen_contents:
                     seen_contents.add(content)
                     deduplicated.append(x)
             
             # Take top k from deduplicated list
             selected = deduplicated[:topk]
             # Construct text
             texts = []
             for i, x in enumerate(selected):
                 content = x.get('content', '').strip()
                 texts.append(f"[{i+1}] {content}")
             retrieved_memory = "\n".join(texts)
             
        mem_prompt = Mem_eval_prompt.format(
            question=question,
            groundtruth_memory=gt_mem_str,
            retrieved_memory=retrieved_memory
        )
        mem_result = run_llm_judge(client, model, mem_prompt)
        if mem_result:
            results.update(mem_result)
            
    return item['id'], results

def extract_user_id_from_filename(filename):
    """Extract user_id from llm metric filename."""
    parts = filename.split('_')
    
    # Find the UUID pattern (should be 36 characters with format 8-4-4-4-12)
    for part in parts:
        # Check if it looks like a UUID
        if len(part) == 36 and part.count('-') == 4:
            # Verify format: 8-4-4-4-12
            uuid_parts = part.split('-')
            if (len(uuid_parts) == 5 and 
                len(uuid_parts[0]) == 8 and 
                len(uuid_parts[1]) == 4 and 
                len(uuid_parts[2]) == 4 and 
                len(uuid_parts[3]) == 4 and 
                len(uuid_parts[4]) == 12):
                return part
    return None

def main(args):
    print(f"Loading generation results from {args.gen_file}...")
    with open(args.gen_file, 'r') as f:
        gen_data = json.load(f)
    file_name = Path(args.gen_file).stem

    user_id = extract_user_id_from_filename(file_name)

    with open(args.qa_file, 'r') as f:
        benchmark_data = json.load(f)
    benchmark_data = next((user for user in benchmark_data if user['person_id'] == user_id), None)
    assert benchmark_data is not None, f"User ID {user_id} not found in benchmark data."
    person_name = benchmark_data['person_name']

    gt_map = load_ground_truth(benchmark_data)
    print(f"Found {len(gt_map)} QA items with ground truth.")


    # Normalize gen_data to list of items
    if isinstance(gen_data, dict):
        # Format is {id: {item_dict}}
        # But we need to make sure 'id' is in the item dict
        gen_items = []
        for qid, item in gen_data.items():
            if not isinstance(item, dict):
                continue
            if 'id' not in item:
                item['id'] = qid
            gen_items.append(item)
        gen_data = gen_items
    
    # Choice Accuracy
    correct_count = 0
    total_choice_q = 0
    
    items_to_judge = []
    
    print("Calculating Choice Accuracy...")
    for item in gen_data:
        # Check choice accuracy
        if item.get('choices'):
            total_choice_q += 1
            pred = item.get('predicted_choice_id', '').strip()
            correct = item.get('correct_choice_id', '').strip()
            # Simple normalization
            if pred and correct and pred.upper() == correct.upper():
                correct_count += 1
        
        if (args.eval_qa or args.eval_mem != 'none') and item['id'] in gt_map:
            items_to_judge.append(item)
    
    acc_summary = {}
    if total_choice_q > 0:
        acc = correct_count / total_choice_q
        print(f"Choice Accuracy: {acc:.4f} ({correct_count}/{total_choice_q})")
        acc_summary['choice_accuracy'] = acc
        acc_summary['correct_count'] = correct_count
        acc_summary['total_choice_questions'] = total_choice_q
    else:
        print("Choice Accuracy: N/A (No choice questions found)")
        acc_summary['choice_accuracy'] = 0.0

    # LLM Judge
    llm_metrics_summary = {}
    detailed_results = {item['id']: {} for item in items_to_judge}

    # Initialize normalized_summary just in case loop is skipped
    normalized_summary = {}

    if args.eval_qa or args.eval_mem != 'none':
        # print(f"\nRunning LLM Judge on {len(items_to_judge)} items with {args.max_workers} threads...")
        # print(f"Eval Mem Type: {args.eval_mem}")
        
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        
        llm_metrics = {
            "QA_consistency_score": [],
            "Mem_recall": [],
            "Mem_helpful_score": []
        }
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(evaluate_single_item, item, gt_map[item['id']], client, args.model_name, args.eval_qa, args.eval_mem, args.topk): item['id'] 
                for item in items_to_judge
            }
            
            for future in tqdm(as_completed(futures), total=len(items_to_judge), desc="LLM Judge", disable=True):
                qid, res = future.result()
                if res:
                    detailed_results[qid] = res
                    for k, v in res.items():
                        if k in llm_metrics and isinstance(v, (int, float)):
                            llm_metrics[k].append(v)
        
        print("\nLLM Judge Metrics:")
        for m, vals in llm_metrics.items():
            if vals:
                avg = np.mean(vals)
                print(f"  {m}: {avg:.4f}")
                llm_metrics_summary[m] = avg
                
                # Special handling for QA_consistency_score distribution
                if m == "QA_consistency_score":
                    dist = {i: vals.count(i) for i in range(4)}
                    print(f"  {m} Distribution: {dist}")
                    llm_metrics_summary[f"{m}_distribution"] = dist
                    
                    hallucination_rate = dist.get(0, 0) / len(vals)
                    perfect_rate = dist.get(3, 0) / len(vals)
                    
                    print(f"  {m} Hallucination Rate: {hallucination_rate:.4f}")
                    print(f"  {m} Perfect Rate: {perfect_rate:.4f}")
                    
                    llm_metrics_summary["qa_hallucination_rate"] = hallucination_rate
                    llm_metrics_summary["qa_perfect_rate"] = perfect_rate
                    llm_metrics_summary["qa_score_distribution"] = dist
            else:
                llm_metrics_summary[m] = 0.0

        # Calculate Normalized Metrics
        if "QA_consistency_score" in llm_metrics_summary:
            normalized_summary["QA_consistency_score_norm"] = llm_metrics_summary["QA_consistency_score"] / 3.0
        if "Mem_helpful_score" in llm_metrics_summary:
            normalized_summary["Mem_helpful_score_norm"] = llm_metrics_summary["Mem_helpful_score"] / 2.0
        if "Mem_recall" in llm_metrics_summary:
            normalized_summary["Mem_recall_norm"] = llm_metrics_summary["Mem_recall"]

        print("\nNormalized Metrics (0-1):")
        for k, v in normalized_summary.items():
            print(f"  {k}: {v:.4f}")
    
    # Save detailed results if needed
    dir_name = os.path.dirname(args.gen_file)
    if not args.out_file:
        output_dir = os.path.join(dir_name, 'llm_judge_detailed_results', )
        os.makedirs(output_dir, exist_ok=True)

        args.out_file = os.path.join(output_dir, f"{person_name}_llm_judge_{file_name}")


    output_data = {
        **acc_summary,
        "llm_metrics_avg": llm_metrics_summary,
        "normalized_metrics": normalized_summary,
        "detailed_llm_results": detailed_results
    }
    with open(args.out_file, 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"\nDetailed evaluation results saved to {args.out_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
