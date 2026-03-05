#!/usr/bin/env python3
"""
Oracle version of retrieval evaluation - no actual retrieval or embedding.
Simply uses the ground truth media_ids from qa_items to form ranking results.
"""

import sys
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime
import multiprocessing as mp


def process_user(user_data, save_path, args):
    """
    Process user data using oracle approach: directly use media_ids from qa_items
    as the ranking result without any retrieval.
    
    Args:
        user_data: User data containing context and qa_items
        save_path: Path to save results
        args: Command line arguments
    """
    user_name = user_data["person_name"]
    user_id = user_data["person_id"]
    
    # Build a mapping from context id to context data for quick lookup
    context_map = {ctx['id']: ctx for ctx in user_data['context']}
    
    print(f"🔍 Processing {len(user_data['context'])} contexts and {len(user_data['qa_items'])} questions (Oracle mode - no retrieval)...")
    
    results = {}
    
    # Process each question
    for question_dict in tqdm(user_data['qa_items'], desc=f"Processing questions for {user_name}"):
        question_id = question_dict['id']
        question = question_dict['question']
        question_time_str = question_dict['question_time']
        media_ids = question_dict.get('media_ids', [])
        
        if not media_ids:
            raise ValueError(f"No media_ids found for question {question_id} in user {user_name}")
        
        # Build ranked items directly from media_ids (oracle ranking)
        ranked_items = []

        if args.use_statement_only:
            for rank, evidence in enumerate(question_dict['evidence']):
                media_id = evidence['related_media_id'][0]
                timestamp = context_map[media_id]['event_date']
                ranked_items.append({
                    'res_type': 'statement',
                    'rank': rank,
                    'chunk_id': media_id,
                    'content': evidence['statement'],
                    'timestamp': timestamp
                })
        else:
            for rank, media_id in enumerate(media_ids):
                if media_id not in context_map:
                    raise ValueError(f"media_id {media_id} not found in context for question {question_id} in user {user_name}")
                
                ctx = context_map[media_id]
                timestamp = ctx['event_date']
                content = ctx['content']
                
                # Add the chunk (original content)
                ranked_items.append({
                    'res_type': 'chunk',
                    'rank': rank,
                    'chunk_id': media_id,
                    'content': content,
                    'timestamp': timestamp
                })

            if args.add_statement:
                # Find corresponding statements for this media_id
                for evidence in question_dict['evidence']:
                    media_id = evidence['related_media_id'][0]
                    timestamp = context_map[media_id]['event_date']
                    ranked_items.append({
                        'res_type': 'statement',
                        'rank': rank,
                        'chunk_id': media_id,
                        'content': evidence['statement'],
                        'timestamp': timestamp
                    })
        
        results[question_id] = {
            'question': question,
            'question_time': question_time_str,
            'num_contexts_processed': len(media_ids),
            'ranked_items': ranked_items
        }
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved oracle results for {user_name} to {save_path}")
    print(f"📊 Statistics: {len(results)} questions processed")
    return results


def process_user_wrapper(user_data_with_args):
    """
    Wrapper function for multiprocessing.
    """
    user_data, args, output_dir, timestamp = user_data_with_args
    user_name = user_data.get('person_name', 'unknown')
    user_id = user_data.get('person_id', 'unknown')
    
    # Generate output filename
    output_filename = f"{args.outfile_prefix}_{user_id}_oracle.json"
    save_path = os.path.join(output_dir, output_filename)
    
    # Check if already processed (resume functionality)
    if args.resume and os.path.exists(save_path):
        print(f"⏭️  Skipping {user_name} - already processed at {save_path}")
        with open(save_path, 'r', encoding='utf-8') as f:
            user_results = json.load(f)
        return (user_id, user_results, save_path, None)
    
    print(f"\n{'='*60}")
    print(f"Processing user: {user_name} ({user_id}) [Oracle Mode]")
    print(f"Context items: {len(user_data.get('context', []))}")
    print(f"QA items: {len(user_data.get('qa_items', []))}")
    print(f"{'='*60}\n")
    
    try:
        user_results = process_user(user_data, save_path, args)
        return (user_id, user_results, save_path, None)
    except Exception as e:
        error_msg = f"❌ Error processing {user_name}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return (user_id, None, save_path, error_msg)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Oracle Retrieval (No Embedding/Retrieval)")
    parser.add_argument('--in_file', type=str, default='data/new_data/all_users_benchmark_en.json', help='Input JSON file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/new_data/oracle/', help='Output directory')
    parser.add_argument('--outfile_prefix', type=str, default='default', help='Output file prefix')
    
    parser.add_argument('--use_statement_only', action='store_true', default=False, help='Use only statement-type context items to formulate ranked items')
    parser.add_argument('--add_statement', action='store_true', default=False, help='Add statement-type context items to ranked items')
    
    # Hardware configuration
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    
    # Resume configuration
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoints and skip already-processed samples',
                       default=True)
    
    args = parser.parse_args()
    assert not (args.use_statement_only and args.add_statement), "Cannot use both --use_statement_only and --add_statement at the same time."
    
    # Load input data
    print(f"Loading data from {args.in_file}...")
    with open(args.in_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine if data is single user or list of users
    if isinstance(data, dict) and 'person_name' in data:
        # Single user format
        users_data = [data]
    elif isinstance(data, list):
        # Multiple users format
        users_data = data
    else:
        raise ValueError("Input data format not recognized. Expected single user dict or list of users.")
    
    print(f"Found {len(users_data)} user(s) to process")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Number of workers: {args.num_workers}")
    
    # Generate a single timestamp for all users
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process each user
    all_results = {}
    
    if args.num_workers > 1:
        # Multiprocessing mode
        print(f"\n🚀 Using multiprocessing with {args.num_workers} workers")
        
        user_args = [
            (user_data, args, output_dir, timestamp)
            for user_data in users_data
        ]
        
        # Process users in parallel
        with mp.Pool(processes=args.num_workers) as pool:
            results_list = pool.map(process_user_wrapper, user_args)
        
        # Collect results
        for user_id, user_results, save_path, error_msg in results_list:
            if user_results is not None:
                all_results[user_id] = user_results
            if error_msg:
                print(error_msg)
    else:
        # Sequential processing
        print(f"\n📝 Sequential processing (single worker)")
        
        for user_data in users_data:
            user_id, user_results, save_path, error_msg = process_user_wrapper(
                (user_data, args, output_dir, timestamp)
            )
            if user_results is not None:
                all_results[user_id] = user_results
            if error_msg:
                print(error_msg)
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"Processed {len(all_results)} user(s)")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
