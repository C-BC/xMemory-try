import json
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.expanduser("~/playground/AgentMem/clonemem_adaptation")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, 
                        default='checkpoints/new_data/mem0_openai/default_9fdf4fb7-c6d8-4fb1-9e01-581d5b85d84a_openai.json',
                        help='Input file containing retrieval results (JSON format)')
    parser.add_argument('--qa_file', type=str, 
                        default=f'data/new_data/all_users_benchmark_en.json',
                        help='Input file containing QA items and ground truth')
    parser.add_argument('--out_file', type=str, default=None,
                        help='Output file to save detailed metrics (optional)')
    return parser.parse_args()


def load_ground_truth(data):
    """
    Load ground truth from QA file.
    Returns: dict { qa_id: [list of evidence_ref objects] }
    """
    qa_map = {}

    qa_items = data.get('qa_items', [])
    for item in qa_items:
        qa_id = item.get('id')
        evidence_refs = item.get('evidence', [])
        qa_map[qa_id] = evidence_refs
    return qa_map

def compute_metrics_for_k(retrieved_chunk_ids, evidence_refs, k):
    """
    Compute metrics for a single query at a specific K.
    retrieved_chunk_ids: list of chunk IDs ordered by rank
    evidence_refs: list of evidence objects from ground truth
    """
    # Cut off at K
    top_k_ids = set(retrieved_chunk_ids[:k])
    
    # Pre-calculate hits for each evidence
    evidence_status = [] # List of dicts { 'all_found': bool, 'any_found': bool }
    
    all_gt_media_ids = set()
    
    for ev in evidence_refs:
        media_ids = ev.get('related_media_id', [])
        if not media_ids:
            continue
            
        # Add to flat set
        for mid in media_ids:
            all_gt_media_ids.add(mid)
            
        # Check conditions
        found_count = sum(1 for mid in media_ids if mid in top_k_ids)
        
        evidence_status.append({
            'all_found': found_count == len(media_ids),
            'any_found': found_count > 0
        })

    if not evidence_status:
        # If no evidence refs, what should be the score?
        # Assuming 0 if no evidence to retrieve, or N/A. 
        # Usually QA items have evidence. If not, maybe return 0.
        if not all_gt_media_ids:
             return 0.0, 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # 1. Recall-Flat
    # Proportion of all unique ground truth media IDs found
    flat_found_count = sum(1 for mid in all_gt_media_ids if mid in top_k_ids)
    recall_flat = flat_found_count / len(all_gt_media_ids) if all_gt_media_ids else 0.0

    # 2. Recall-All-All: All evidences must have ALL media IDs found
    recall_all_all = 1.0 if all(e['all_found'] for e in evidence_status) else 0.0

    # 3. Recall-All-Any: All evidences must have AT LEAST ONE media ID found
    recall_all_any = 1.0 if all(e['any_found'] for e in evidence_status) else 0.0

    # 4. Recall-Any-All: At least one evidence must have ALL media IDs found
    recall_any_all = 1.0 if any(e['all_found'] for e in evidence_status) else 0.0

    # 5. Recall-Any-Any: At least one evidence must have AT LEAST ONE media ID found
    recall_any_any = 1.0 if any(e['any_found'] for e in evidence_status) else 0.0

    return recall_all_all, recall_all_any, recall_any_all, recall_any_any, recall_flat

def main(args):
    file_name = Path(args.in_file).stem
    user_id = file_name.split('_')[1]
    print(f"Processing for user_id: {user_id}")

    with open(args.qa_file, 'r') as f:
        benchmark_data = json.load(f)

    benchmark_data = next((user for user in benchmark_data if user['person_id'] == user_id), None)
    assert benchmark_data is not None, f"User ID {user_id} not found in benchmark data."
    person_name = benchmark_data['person_name']

    gt_map = load_ground_truth(benchmark_data)
    print(f"Found {len(gt_map)} QA items with ground truth.")

    print(f"Loading retrieval results from {args.in_file}...")
    with open(args.in_file, 'r') as f:
        retrieval_results = json.load(f)
    if 'results' in retrieval_results:
        retrieval_results = retrieval_results['results']

    # Metrics storage
    ks = [10, 20]
    metric_names = ['recall_all_all', 'recall_all_any', 'recall_any_all', 'recall_any_any', 'recall_flat']
    
    # Initialize accumulators
    results = {k: {m: [] for m in metric_names} for k in ks}

    count_evaluated = 0

    for qa_id, result_obj in tqdm(retrieval_results.items(), desc="Evaluating"):
        if qa_id not in gt_map:
            # It might be that retrieval results contain items not in the specific QA file 
            # (if we ran on a subset or different set). 
            # Or vice versa.
            continue
            
        evidence_refs = gt_map[qa_id]
        if not evidence_refs:
             continue
        
        # Extract ranked chunk IDs
        ranked_items = result_obj.get('ranked_items', [])
        # Filter for chunks only
        ranked_chunk_ids = [
            item['chunk_id'] 
            for item in ranked_items 
            if item.get('res_type') == 'chunk' and 'chunk_id' in item
        ]
        
        # Deduplicate chunk_ids (keep first occurrence only)
        # This is important when using subquestion decomposition where same chunk may appear multiple times from different subquestions
        seen = set()
        deduplicated_chunk_ids = []
        for chunk_id in ranked_chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduplicated_chunk_ids.append(chunk_id)
        ranked_chunk_ids = deduplicated_chunk_ids
        
        # Calculate for each K
        for k in ks:
            r_aa, r_aa_ny, r_ny_aa, r_ny_ny, r_flat = compute_metrics_for_k(ranked_chunk_ids, evidence_refs, k)
            
            results[k]['recall_all_all'].append(r_aa)
            results[k]['recall_all_any'].append(r_aa_ny)
            results[k]['recall_any_all'].append(r_ny_aa)
            results[k]['recall_any_any'].append(r_ny_ny)
            results[k]['recall_flat'].append(r_flat)

        count_evaluated += 1

    print("-" * 30)
    print(f"Evaluated {count_evaluated} queries.")
    print("-" * 30)

    # Print Averages
    print("Average Metrics:")
    final_summary = {}
    
    for k in ks:
        print(f"\n@K={k}:")
        k_summary = {}
        for m in metric_names:
            vals = results[k][m]
            if vals:
                avg = np.mean(vals)
                print(f"  {m}: {avg:.4f}")
                k_summary[m] = avg
            else:
                print(f"  {m}: N/A")
                k_summary[m] = 0.0
        final_summary[f"k{k}"] = k_summary

    if not args.out_file:
        output_dir = os.path.dirname(args.in_file)
        output_dir = os.path.join(output_dir, 'recall_metrics')
        os.makedirs(output_dir, exist_ok=True)
        
        args.out_file = os.path.join(output_dir, f"recall_metrics_{person_name}_{file_name}.json")

    with open(args.out_file, 'w') as f:
        json.dump(final_summary, f, indent=4)
    print(f"\nSummary saved to {args.out_file} for user: {person_name} (ID: {user_id})")

if __name__ == "__main__":
    args = parse_args()
    main(args)
