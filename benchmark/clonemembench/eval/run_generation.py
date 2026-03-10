import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def construct_evidence_text(ranked_items, evidence_type='chunk', top_k=5):
    """
    Construct evidence text from ranked items.
    
    Args:
        ranked_items: List of ranked retrieval items
        evidence_type: Type of evidence to use (e.g., 'chunk', 'extract_context')
        top_k: Number of top evidence items to use
    
    Returns:
        Formatted evidence text string
    
    Behavior:
        - If no subquestions: selects top-k items sequentially
        - If subquestions present: round-robin selection from each subquestion with deduplication by chunk_id
    """
    # Filter items by evidence type (support both 'chunk' and 'memory')
    if evidence_type in ('chunk', 'memory'):
        filtered = [item for item in ranked_items if item.get('res_type') in ('chunk', 'memory')]
    else:
        filtered = [item for item in ranked_items if item.get('res_type') == evidence_type]
    
    # Check if items have subquestion information
    has_subquestions = any('subquestion_id' in item and item['subquestion_id'] > 0 for item in filtered)
    
    if not has_subquestions:
        # No subquestions: just take top-k sequentially
        selected = filtered[:top_k]
    else:
        # Group items by subquestion_id
        print(f"  Detected subquestions in evidence items, applying round-robin selection with deduplication by chunk_id")
        subq_groups = {}
        for item in filtered:
            subq_id = item['subquestion_id']
            if subq_id not in subq_groups:
                subq_groups[subq_id] = []
            subq_groups[subq_id].append(item)
        
        # Round-robin selection with deduplication by chunk_id
        selected = []
        seen_chunks = set()
        subq_ids = sorted(subq_groups.keys())
        max_items_per_subq = max(len(items) for items in subq_groups.values()) if subq_groups else 0
        
        for round_idx in range(max_items_per_subq):
            for subq_id in subq_ids:
                if round_idx < len(subq_groups[subq_id]):
                    item = subq_groups[subq_id][round_idx]
                    chunk_id = item.get('chunk_id')
                    
                    # Skip if we've already seen this chunk_id
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        selected.append(item)
                        if len(selected) >= top_k:
                            break
            if len(selected) >= top_k:
                break
    
    # Format evidence texts
    evidence_texts = []
    for i, item in enumerate(selected):
        content = item.get('content', '').strip()
        # Add metadata if needed
        if item.get('res_type') == 'entity':
            content = f"{item['entity_name']}: {content}"
        
        # Optionally include subquestion info in output
        subq_info = ""
        if has_subquestions and 'subquestion' in item:
            subq_info = f" [from subQ{item.get('subquestion_id', 0)}]"
        
        evidence_texts.append(f"---- idx {i+1}{subq_info} ----\n{content}")
    
    return "\n\n".join(evidence_texts)

def construct_answer_prompt(question, evidence_text, user_name):
    return f"""You are playing the role of {user_name}, answering some questions on his behalf. Your answers must be strictly based on the provided reference memories (from {user_name} himself/herself).

Memories:
{evidence_text}

Question: {question}

Answer:"""


def construct_subquestion_prompt(subquestion, evidence_text, user_name):
    """Construct prompt for answering a single subquestion."""
    return f"""You are playing the role of {user_name}, answering some questions on his behalf. Your answers must be strictly based on the provided reference memories (from {user_name} himself/herself).

Memories:
{evidence_text}

Sub-question: {subquestion}

Answer (be concise and factual):"""


def construct_final_answer_from_subquestions(question, subquestion_data, user_name):
    """
    Construct prompt for final answer using subquestions and their answers.
    
    Args:
        question: Original question
        subquestion_data: List of dicts with 'subquestion', 'evidence', 'answer'
        user_name: User's name
    """
    subq_context = []
    for i, data in enumerate(subquestion_data, 1):
        subq_context.append(f"""Sub-question {i}: {data['subquestion']}

Evidence for sub-question {i}:
{data['evidence']}

Answer to sub-question {i}: {data['answer']}""")
    
    combined_context = "\n\n" + "="*60 + "\n\n".join(subq_context)
    
    return f"""You are playing the role of {user_name}, answering some questions on his behalf.

Below are several sub-questions that were asked to gather relevant information, along with their evidence and answers:

{combined_context}

Original Question: {question}

Based on the sub-questions and their answers above, please provide a comprehensive answer to the original question:"""

def construct_choice_prompt(question, choices, evidence_text, user_name):
    options_str = "\n".join([f"{c['id']}. {c['text']}" for c in choices])
    return f"""You are playing the role of {user_name}, answering some questions on his behalf. Your answers must be strictly based on the provided reference memories (from {user_name} himself/herself). Respond with only the option ID (e.g., A, B, C, D).

Memories:
{evidence_text}

Question: {question}

Options:
{options_str}

Correct Option ID:"""

def process_item(item, retrieval_data, args, client, person_name):
    qid = item['id']
    question = item['question']
    choices = item.get('choices', [])
    
    # Get retrieval results
    ret_result = retrieval_data.get(qid, {})
    ranked_items = ret_result.get('ranked_items', [])
    subquestions = ret_result.get('subquestions', [])
    
    # Check if we should use subquestion-based answering
    has_subquestions = bool(subquestions) and any('subquestion_id' in rank_item for rank_item in ranked_items)
    use_subq_mode = has_subquestions and args.answer_subquestions_separately
    
    if use_subq_mode:
        # Mode: Answer each subquestion separately, then combine for final answer
        print(f"  Processing {qid} with subquestion-based answering ({len(subquestions)} subquestions)")
        
        # Group ranked items by subquestion_id
        if args.evidence_type in ('chunk', 'memory'):
            filtered = [rank_item for rank_item in ranked_items if rank_item.get('res_type') in ('chunk', 'memory')]
        else:
            filtered = [rank_item for rank_item in ranked_items if rank_item.get('res_type') == args.evidence_type]
        subq_items = {}
        for rank_item in filtered:
            subq_id = rank_item.get('subquestion_id', 0)
            if subq_id not in subq_items:
                subq_items[subq_id] = []
            subq_items[subq_id].append(rank_item)
        
        # Answer each subquestion separately
        subquestion_data = []
        for subq_id, subq_text in enumerate(subquestions):
            # Get top-k items for this subquestion
            subq_ranked = subq_items.get(subq_id, [])
            subq_evidence_items = subq_ranked[:args.top_k]
            
            # Format evidence for this subquestion
            evidence_texts = []
            for i, ev_item in enumerate(subq_evidence_items):
                content = ev_item.get('content', '').strip()
                evidence_texts.append(f"---- idx {i+1} ----\n{content}")
            subq_evidence = "\n\n".join(evidence_texts)
            
            # Generate answer for this subquestion
            subq_answer = ""
            try:
                subq_prompt = construct_subquestion_prompt(subq_text, subq_evidence, person_name)
                subq_completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": subq_prompt}],
                    temperature=0.7
                )
                subq_answer = subq_completion.choices[0].message.content
            except Exception as e:
                print(f"Error answering subquestion {subq_id} for {qid}: {e}")
                subq_answer = "[Error generating answer]"
            
            subquestion_data.append({
                'subquestion': subq_text,
                'evidence': subq_evidence,
                'answer': subq_answer
            })
        
        # Generate final answer using all subquestions and their answers
        final_prompt = construct_final_answer_from_subquestions(question, subquestion_data, person_name)
        gen_answer = ""
        try:
            final_completion = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.7
            )
            gen_answer = final_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating final answer for {qid}: {e}")
        
        # For evidence_used, store the subquestion-based structure
        evidence_text = final_prompt  # Store the full prompt with all subquestions
        
    else:
        # Original mode: Direct answering with combined evidence
        evidence_text = construct_evidence_text(
            ranked_items, 
            args.evidence_type, 
            args.top_k
        )
        
        # Task 1: Generate Answer
        ans_prompt = construct_answer_prompt(question, evidence_text, person_name)
        gen_answer = ""
        try:
            ans_completion = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": ans_prompt}],
                temperature=0.7 
            )
            gen_answer = ans_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer for {qid}: {e}")

    # Task 2: Predict Choice (same for both modes)
    pred_choice = ""
    if choices:
        # For choice prediction, always use combined evidence (simpler)
        if use_subq_mode:
            # Use round-robin combined evidence for choices
            choice_evidence = construct_evidence_text(ranked_items, args.evidence_type, args.top_k)
        else:
            choice_evidence = evidence_text
            
        choice_prompt = construct_choice_prompt(question, choices, choice_evidence, person_name)
        try:
            choice_completion = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": choice_prompt}],
                temperature=0.1
            )
            pred_choice = choice_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error predicting choice for {qid}: {e}")
    
    result = {
        "id": qid,
        "question": question,
        "gold_answer": item.get('answer'),
        "generated_answer": gen_answer,
        "choices": choices,
        "correct_choice_id": item.get('correct_choice_id'),
        "predicted_choice_id": pred_choice,
        "evidence_used": evidence_text
    }
    
    # Add subquestion data if available
    if use_subq_mode:
        result['subquestion_data'] = subquestion_data
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run generation for QA items using retrieval results.")
    parser.add_argument('--input_file', default='data/new_data/all_users_benchmark_en.json', help="Path to the benchmark input file (JSON).")
    parser.add_argument('--retrieval_file', default='', help="Path to the retrieval results file (JSON).")
    parser.add_argument('--output_file', default='', help="Path to save the output results.")
    parser.add_argument('--model_name', default='gpt-4o-mini', help="Model name to use (e.g., gpt-4o-mini).")
    parser.add_argument('--base_url', default='http://localhost:8001/v1', help="Base URL for the API (e.g., for local VLLM).")
    parser.add_argument('--api_key', default=None, help="API Key for the client.")
    parser.add_argument('--evidence_type', default='chunk', help="Type of evidence to use (default: chunk).")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top evidence items to use.")
    parser.add_argument('--answer_subquestions_separately', action='store_true', 
                       help="When subquestions are present, answer each subquestion separately with its own evidence, "
                            "then combine all sub-answers to generate the final answer (default: False, use round-robin combined evidence)")
    parser.add_argument('--max_workers', type=int, default=10, help="Number of concurrent workers (default: 10).")
    
    args = parser.parse_args()
    
    # Setup client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_key = 'EMPTY' if not 'gpt' in args.model_name else api_key
    args.base_url = None if 'gpt' in args.model_name else args.base_url
    
    print(f"Initializing OpenAI client with model={args.model_name}, base_url={args.base_url}")
    print(f"Evidence selection: type={args.evidence_type}, top_k={args.top_k}")
    # OpenAI client is thread-safe
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    # Load data
    print(f"Loading input data from {args.input_file}...")
    benchmark_data = load_data(args.input_file)
    
    print(f"Loading retrieval data from {args.retrieval_file}...")
    retrieval_data = load_data(args.retrieval_file)
    if 'results' in retrieval_data:
        retrieval_data = retrieval_data['results']
    file_name = Path(args.retrieval_file).stem
    user_id = file_name.split('_')[1]

    # find the corresponding user data
    benchmark_data = next((user for user in benchmark_data if user['person_id'] == user_id), None)
    assert benchmark_data is not None, f"User ID {user_id} not found in benchmark data."
    
    person_name = benchmark_data.get('person_name')
    if not person_name:
        raise ValueError("Person name not found in the input data.")
    print(f"Processing for user: {person_name} (ID: {user_id})")
    
    qa_items = benchmark_data.get('qa_items', [])
    print(f"Found {len(qa_items)} QA items.")
    
    # Check if retrieval data contains subquestions
    has_subquestions = False
    num_subquestions = 0
    for qid, data in retrieval_data.items():
        if 'subquestions' in data and data['subquestions']:
            has_subquestions = True
            num_subquestions = len(data['subquestions'])
            break
    
    if has_subquestions:
        print(f"✓ Detected subquestion decomposition in retrieval results ({num_subquestions} subquestions per question)")
        if args.answer_subquestions_separately:
            print(f"  Mode: Answer each subquestion separately (top-{args.top_k} per subquestion), then combine for final answer")
        else:
            print(f"  Mode: Round-robin selection with deduplication by chunk_id (top-{args.top_k} total)")
    else:
        print(f"ℹ No subquestion decomposition detected in retrieval results")
        print(f"  Using sequential top-k selection")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, item, retrieval_data, args, client, person_name): item['id'] for item in qa_items}
        
        for future in tqdm(as_completed(futures), total=len(qa_items), desc="Processing Items"):
            try:
                result = future.result()
                qid = result['id']
                if qid not in retrieval_data:
                    retrieval_data[qid] = {}
                retrieval_data[qid].update(result)
            except Exception as e:
                qid = futures[future]
                print(f"Error processing item {qid}: {e}")
        
    # Save results (retrieval_data with updates)
    output_dir = os.path.dirname(args.retrieval_file)
    output_dir = os.path.join(output_dir, 'generation_results')
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename
    mode_suffix = "_subq_separate" if (has_subquestions and args.answer_subquestions_separately) else ""
    args.output_file = os.path.join(output_dir, os.path.basename(args.output_file)) if args.output_file else os.path.join(output_dir, f"{person_name}_generation_top{args.top_k}{mode_suffix}_{file_name}.json")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
