#!/usr/bin/env python3
"""
xMemory Evaluation on CloneMemBench

This script evaluates xMemory's hierarchical memory system on the CloneMemBench
benchmark. It processes each user's digital traces through xMemory's episodic ->
semantic -> theme pipeline, then uses xMemory's hybrid search for retrieval.

The output format is compatible with CloneMemBench's standard evaluation scripts
(compute_auto_metrics, run_generation, compute_llm_metrics).

Usage:
    python eval_xmemory.py \
        --in_file data/all_users_benchmark_en_100k.json \
        --output_dir checkpoints/xmemory/ \
        --llm_model gpt-4o-mini \
        --embedding_model text-embedding-3-small \
        --search_strategy hybrid \
        --retrieve_k 20
"""

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add xMemory project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in [REPO_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from xMemory import xMemory, MemoryConfig

logger = logging.getLogger(__name__)


def build_xmemory_config(args) -> MemoryConfig:
    """Build xMemory configuration from command-line arguments."""
    storage_path = os.path.join(args.output_dir, "memories")
    chroma_path = os.path.join(storage_path, "chroma_db")

    return MemoryConfig(
        storage_path=storage_path,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        language=args.language,
        # Search configuration
        search_top_k_episodes=args.retrieve_k,
        search_top_k_semantic=args.retrieve_k,
        enable_parallel_search=True,
        # Semantic memory
        enable_semantic_memory=True,
        enable_prediction_correction=True,
        extract_semantic_per_episode=True,
        # Storage backends
        storage_backend="filesystem",
        vector_index_backend="chroma",
        lexical_index_backend="bm25",
        chroma_persist_directory=chroma_path,
        chroma_collection_prefix=f"clonemem_{args.outfile_prefix}",
        # Buffer: treat each digital trace as a complete episode
        buffer_size_min=2,
        buffer_size_max=25,
        # Performance
        semantic_generation_workers=args.semantic_workers,
    )


def ingest_contexts(mem: xMemory, user_id: str, contexts: list, question_time: str) -> int:
    """
    Ingest digital trace contexts into xMemory as messages.

    Only ingest contexts with event_date <= question_time.
    Each context is treated as a complete conversation unit.

    Returns:
        Number of contexts ingested.
    """
    q_time = datetime.fromisoformat(question_time)
    count = 0

    for ctx in contexts:
        ctx_time = datetime.fromisoformat(ctx["event_date"])
        if ctx_time > q_time:
            continue

        # Construct a message pair: system context + user content
        messages = [
            {
                "role": "user",
                "content": ctx["content"],
                "timestamp": ctx["event_date"],
                "metadata": {
                    "context_id": ctx["id"],
                    "medium": ctx.get("medium", "unknown"),
                },
            }
        ]

        mem.add_messages(user_id, messages)
        count += 1

    return count


def search_and_build_ranked_items(
    mem: xMemory,
    user_id: str,
    question: str,
    retrieve_k: int,
    search_method: str,
    context_id_map: dict,
) -> list:
    """
    Search xMemory and convert results to CloneMemBench ranked_items format.

    Args:
        mem: xMemory instance
        user_id: User identifier
        question: Query string
        retrieve_k: Number of results to retrieve
        search_method: Search method (hybrid, vector, bm25)
        context_id_map: Mapping from content substrings to original context IDs

    Returns:
        List of ranked_items dicts compatible with CloneMemBench eval
    """
    results = mem.search(
        user_id,
        question,
        top_k_episodes=retrieve_k,
        top_k_semantic=retrieve_k,
        search_method=search_method,
    )

    ranked_items = []
    seen_chunk_ids = set()
    rank = 0

    # Process episodic results
    for ep in results.get("episodes", []):
        content = ep.get("content", "") or ep.get("title", "")
        chunk_id = _match_context_id(content, ep, context_id_map)

        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            ranked_items.append({
                "res_type": "chunk",
                "rank": rank,
                "chunk_id": chunk_id,
                "content": content,
                "timestamp": ep.get("timestamp", ""),
            })
            rank += 1

    # Process semantic results
    for sem in results.get("semantic", []):
        content = sem.get("content", "")
        chunk_id = _match_context_id(content, sem, context_id_map)

        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            ranked_items.append({
                "res_type": "chunk",
                "rank": rank,
                "chunk_id": chunk_id,
                "content": content,
                "timestamp": sem.get("timestamp", ""),
            })
            rank += 1

    return ranked_items


def _match_context_id(content: str, result_item: dict, context_id_map: dict) -> str:
    """
    Try to match a search result back to the original context ID.

    Strategy:
    1. Check metadata for context_id
    2. Check source_episode_ids or source_messages
    3. Fuzzy match by content overlap
    """
    # Try direct metadata
    metadata = result_item.get("metadata", {})
    if isinstance(metadata, dict):
        ctx_id = metadata.get("context_id")
        if ctx_id:
            return ctx_id

    # Try source references
    for key in ["source_episode_id", "source_episode_ids", "episode_id"]:
        val = result_item.get(key)
        if val:
            if isinstance(val, list):
                val = val[0] if val else None
            if val and val in context_id_map:
                return context_id_map[val]

    # Fuzzy match: find the context whose content best overlaps
    if content:
        best_id = None
        best_overlap = 0
        content_lower = content[:500].lower()
        for ctx_id, ctx_content in context_id_map.items():
            if isinstance(ctx_content, str):
                overlap = _compute_overlap(content_lower, ctx_content[:500].lower())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = ctx_id
        if best_overlap > 0.3:
            return best_id

    return ""


def _compute_overlap(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def process_user(user_data: dict, args) -> dict:
    """
    Process a single user through xMemory and generate retrieval results.

    This follows CloneMemBench's temporal awareness: for each question,
    only contexts before the question_time are available.
    """
    user_name = user_data["person_name"]
    user_id = user_data["person_id"]
    contexts = sorted(user_data["context"], key=lambda x: datetime.fromisoformat(x["event_date"]))
    qa_items = sorted(user_data["qa_items"], key=lambda x: datetime.fromisoformat(x["question_time"]))

    print(f"\n{'='*60}")
    print(f"Processing user: {user_name} ({user_id})")
    print(f"Contexts: {len(contexts)}, Questions: {len(qa_items)}")
    print(f"{'='*60}")

    # Build context_id -> content map for matching
    context_id_map = {ctx["id"]: ctx["content"] for ctx in contexts}

    # Build xMemory config with user-specific paths
    config = build_xmemory_config(args)
    config.storage_path = os.path.join(args.output_dir, "memories", user_id)
    config.chroma_persist_directory = os.path.join(config.storage_path, "chroma_db")
    config.chroma_collection_prefix = f"clonemem_{user_id[:8]}"

    results = {}
    added_context_ids = set()
    context_pointer = 0

    # Initialize xMemory for this user
    mem = xMemory(config=config)

    try:
        for question_dict in tqdm(qa_items, desc=f"Questions for {user_name}"):
            question_id = question_dict["id"]
            question = question_dict["question"]
            question_time_str = question_dict["question_time"]
            question_time = datetime.fromisoformat(question_time_str)

            # Add all contexts up to question_time
            new_contexts = 0
            while context_pointer < len(contexts):
                ctx = contexts[context_pointer]
                ctx_time = datetime.fromisoformat(ctx["event_date"])

                if ctx_time > question_time:
                    break

                ctx_id = ctx["id"]
                if ctx_id not in added_context_ids:
                    messages = [
                        {
                            "role": "user",
                            "content": ctx["content"],
                            "timestamp": ctx["event_date"],
                            "metadata": {
                                "context_id": ctx_id,
                                "medium": ctx.get("medium", "unknown"),
                            },
                        }
                    ]
                    mem.add_messages(user_id, messages)
                    added_context_ids.add(ctx_id)
                    new_contexts += 1

                context_pointer += 1

            # Force flush any buffered messages and wait for semantic processing
            if new_contexts > 0:
                try:
                    mem.flush(user_id)
                except Exception:
                    pass  # flush may raise if buffer is empty
                mem.wait_for_semantic(user_id, timeout=60.0)

            # Search
            ranked_items = search_and_build_ranked_items(
                mem=mem,
                user_id=user_id,
                question=question,
                retrieve_k=args.retrieve_k,
                search_method=args.search_strategy,
                context_id_map=context_id_map,
            )

            results[question_id] = {
                "question": question,
                "question_time": question_time_str,
                "num_contexts_processed": len(added_context_ids),
                "ranked_items": ranked_items,
            }

    finally:
        mem.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="xMemory Evaluation on CloneMemBench")

    # Data & Output
    parser.add_argument("--in_file", type=str, required=True,
                        help="Input JSON file (output of prepare_data.py)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/xmemory/",
                        help="Output directory for results")
    parser.add_argument("--outfile_prefix", type=str, default="xmemory",
                        help="Output file prefix")

    # Model Configuration
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="LLM model for xMemory (boundary detection, episode/semantic generation)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                        help="Embedding model for xMemory vector search")

    # Search Configuration
    parser.add_argument("--search_strategy", type=str, default="hybrid",
                        choices=["hybrid", "vector", "bm25"],
                        help="Search strategy to use")
    parser.add_argument("--retrieve_k", type=int, default=20,
                        help="Number of results to retrieve")

    # Language
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                        help="Language for processing")

    # Performance
    parser.add_argument("--semantic_workers", type=int, default=8,
                        help="Number of semantic generation workers")

    # Resume
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoints")

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.in_file}...")
    with open(args.in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "person_name" in data:
        users_data = [data]
    elif isinstance(data, list):
        users_data = data
    else:
        raise ValueError("Input data format not recognized.")

    print(f"Found {len(users_data)} user(s) to process")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"LLM model: {args.llm_model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Search strategy: {args.search_strategy}")
    print(f"Retrieve K: {args.retrieve_k}")

    all_results = {}

    for user_data in users_data:
        user_id = user_data["person_id"]
        user_name = user_data["person_name"]

        # Output filename
        output_filename = f"{args.outfile_prefix}_{user_id}_{args.search_strategy}.json"
        save_path = os.path.join(args.output_dir, output_filename)

        # Check if already processed
        if args.resume and os.path.exists(save_path):
            print(f"Skipping {user_name} - already processed at {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                user_results = json.load(f)
            all_results[user_id] = user_results
            continue

        try:
            user_results = process_user(user_data, args)

            # Save results
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(user_results, f, ensure_ascii=False, indent=2)
            print(f"Saved results for {user_name} to {save_path}")

            all_results[user_id] = user_results
        except Exception as e:
            print(f"Error processing {user_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processing complete! {len(all_results)} user(s) processed.")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
