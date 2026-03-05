import os
import json
from pathlib import Path
import numpy as np


root = 'checkpoints/new_data/flatten/generation_results/llm_judge_detailed_results'
pattern = 'top20'
pattern2 = 'merge-no_chunk'

RECOMPUTE_CONSISTENCY = True  # recompute qa_consistency and map it: 0/1 -> 0, 2 -> 1, 3 -> 2

def compute_average_metrics(root_dir, pattern='', pattern2=''):
    """
    Compute average metrics from all JSON files in the directory.
    """

    root_path = Path(root_dir)
    
    # Find all files matching the pattern
    if pattern:
        files = list(root_path.glob(f'*{pattern}*'))
        if pattern2:
            files = [f for f in files if pattern2 in f.name]
    else:
        files = list(root_path.iterdir())
    
    # Filter for files (not directories)
    files = [f for f in files if f.is_file()]
    
    if not files:
        print(f"No files found in {root_dir}")
        return
    
    assert len(files) == 10, f"Expected 10 files, but found {len(files)}"
    
    # Collect metrics from all files
    choice_accuracies = []
    qa_consistency_scores = []
    mem_recalls = []
    mem_helpful_scores = []
    qa_hallucination_rates = []
    qa_perfect_rates = []
    qa_score_distributions = {0: [], 1: [], 2: [], 3: []}
    
    qa_consistency_norm = []
    mem_helpful_norm = []
    mem_recall_norm = []
    
    # Store individual file results for printing
    individual_results = []
    
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Store individual results
        file_result = {'filename': file_path.name}
        
        # Extract choice accuracy
        if 'choice_accuracy' in data:
            choice_accuracies.append(data['choice_accuracy'])
            file_result['choice_accuracy'] = data['choice_accuracy']
        
        # Extract LLM metrics
        llm_metrics = data.get('llm_metrics_avg', {})

        if RECOMPUTE_CONSISTENCY:
            qa_consistency_distr = llm_metrics['QA_consistency_score_distribution']
            total = qa_consistency_distr['2'] + 2 * qa_consistency_distr['3']
            qa_consistency_score = total / sum(qa_consistency_distr.values())
            qa_consistency_scores.append(qa_consistency_score)
            file_result['QA_consistency_score'] = qa_consistency_score
        else:    
            qa_consistency_scores.append(llm_metrics['QA_consistency_score'])
            file_result['QA_consistency_score'] = llm_metrics['QA_consistency_score']
    
        mem_recalls.append(llm_metrics['Mem_recall'])
        file_result['Mem_recall'] = llm_metrics['Mem_recall']
    
        mem_helpful_scores.append(llm_metrics['Mem_helpful_score'])
        file_result['Mem_helpful_score'] = llm_metrics['Mem_helpful_score']
    
        qa_hallucination_rates.append(llm_metrics['qa_hallucination_rate'])
        file_result['qa_hallucination_rate'] = llm_metrics['qa_hallucination_rate']
    
        qa_perfect_rates.append(llm_metrics['qa_perfect_rate'])
        file_result['qa_perfect_rate'] = llm_metrics['qa_perfect_rate']
    
        dist = llm_metrics['qa_score_distribution']
        for score in range(4):
            if str(score) in dist:
                qa_score_distributions[score].append(dist[str(score)])
        
        # Extract normalized metrics
        norm_metrics = data.get('normalized_metrics', {})
        
        if 'QA_consistency_score_norm' in norm_metrics:
            if RECOMPUTE_CONSISTENCY:
                norm_metrics['QA_consistency_score_norm'] = file_result['QA_consistency_score'] / 2.0
            qa_consistency_norm.append(norm_metrics['QA_consistency_score_norm'])
            file_result['QA_consistency_score_norm'] = norm_metrics['QA_consistency_score_norm']
        
        if 'Mem_helpful_score_norm' in norm_metrics:
            mem_helpful_norm.append(norm_metrics['Mem_helpful_score_norm'])
            file_result['Mem_helpful_score_norm'] = norm_metrics['Mem_helpful_score_norm']
        
        if 'Mem_recall_norm' in norm_metrics:
            mem_recall_norm.append(norm_metrics['Mem_recall_norm'])
            file_result['Mem_recall_norm'] = norm_metrics['Mem_recall_norm']
        
        individual_results.append(file_result)
                
    
    # Print individual file results first
    print("="*60)
    print("INDIVIDUAL FILE METRICS")
    print("="*60)
    
    for result in individual_results:
        print(f"\nFile: {result['filename']}")
        if 'choice_accuracy' in result:
            print(f"  Choice Accuracy: {result['choice_accuracy']:.4f}")
        print(f"  QA_consistency_score: {result['QA_consistency_score']:.4f}")
        print(f"  Mem_recall: {result['Mem_recall']:.4f}")
        print(f"  Mem_helpful_score: {result['Mem_helpful_score']:.4f}")
        print(f"  qa_hallucination_rate: {result['qa_hallucination_rate']:.4f}")
        print(f"  qa_perfect_rate: {result['qa_perfect_rate']:.4f}")
        if 'QA_consistency_score_norm' in result:
            print(f"  QA_consistency_score_norm: {result['QA_consistency_score_norm']:.4f}")
        if 'Mem_helpful_score_norm' in result:
            print(f"  Mem_helpful_score_norm: {result['Mem_helpful_score_norm']:.4f}")
        if 'Mem_recall_norm' in result:
            print(f"  Mem_recall_norm: {result['Mem_recall_norm']:.4f}")
    
    # Print results in the same format as compute_llm_metrics_for_clonemem.py
    print("\n" + "="*60)
    print("AVERAGE METRICS ACROSS ALL FILES")
    print("="*60)
    
    if choice_accuracies:
        avg_choice_acc = np.mean(choice_accuracies)
        print(f"\nChoice Accuracy: {avg_choice_acc:.4f}")
    else:
        print("\nChoice Accuracy: N/A (No choice questions found)")
    
    print("\nLLM Judge Metrics:")
    
    if qa_consistency_scores:
        avg_qa_score = np.mean(qa_consistency_scores)
        print(f"  QA_consistency_score: {avg_qa_score:.4f}")
        
        # Compute average distribution
        avg_dist = {}
        total_items = sum(len(qa_score_distributions[i]) for i in range(4))
        if total_items > 0:
            for score in range(4):
                if qa_score_distributions[score]:
                    avg_dist[score] = int(np.sum(qa_score_distributions[score]))
            print(f"  QA_consistency_score Distribution: {avg_dist}")
            
            # Compute average rates
            if qa_hallucination_rates:
                avg_hal_rate = np.mean(qa_hallucination_rates)
                print(f"  QA_consistency_score Hallucination Rate: {avg_hal_rate:.4f}")
            
            if qa_perfect_rates:
                avg_perfect_rate = np.mean(qa_perfect_rates)
                print(f"  QA_consistency_score Perfect Rate: {avg_perfect_rate:.4f}")
    
    if mem_recalls:
        avg_mem_recall = np.mean(mem_recalls)
        print(f"  Mem_recall: {avg_mem_recall:.4f}")
    
    if mem_helpful_scores:
        avg_mem_helpful = np.mean(mem_helpful_scores)
        print(f"  Mem_helpful_score: {avg_mem_helpful:.4f}")
    
    print("\nNormalized Metrics (0-1):")
    
    if qa_consistency_norm:
        avg_qa_norm = np.mean(qa_consistency_norm)
        print(f"  QA_consistency_score_norm: {avg_qa_norm:.4f}")
    
    if mem_helpful_norm:
        avg_mem_helpful_norm = np.mean(mem_helpful_norm)
        print(f"  Mem_helpful_score_norm: {avg_mem_helpful_norm:.4f}")
    
    if mem_recall_norm:
        avg_mem_recall_norm = np.mean(mem_recall_norm)
        print(f"  Mem_recall_norm: {avg_mem_recall_norm:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    compute_average_metrics(root, pattern, pattern2)

