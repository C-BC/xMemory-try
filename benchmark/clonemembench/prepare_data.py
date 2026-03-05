#!/usr/bin/env python3
"""
Prepare CloneMemBench released data for evaluation.

The released data uses field names like `questions`, `digital_trace_ids`,
while the evaluation scripts expect `qa_items`, `media_ids`, `related_media_id`.
This script converts the data into the eval-compatible format.

Usage:
    python prepare_data.py --context_len 100k --output_file data/all_users_benchmark_en.json
    python prepare_data.py --context_len 500k --output_file data/all_users_benchmark_en_500k.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a question from the released schema to the eval schema."""
    converted = {
        "id": q["id"],
        "question": q["question"],
        "question_type": q.get("question_type", ""),
        "question_time": q["question_time"],
        "answer": q.get("answer", ""),
        "dimension": q.get("dimension", ""),
        # Map digital_trace_ids -> media_ids
        "media_ids": q.get("digital_trace_ids", q.get("media_ids", [])),
        "choices": q.get("choices", []),
        "correct_choice_id": q.get("correct_choice_id", ""),
    }

    # Convert evidence: digital_trace_ids -> related_media_id
    evidence_list = []
    for ev in q.get("evidence", []):
        evidence_list.append({
            "statement": ev.get("statement", ""),
            "related_media_id": ev.get("digital_trace_ids", ev.get("related_media_id", [])),
        })
    converted["evidence"] = evidence_list

    return converted


def convert_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single user's data from released schema to eval schema."""
    # Determine the question field name
    questions = user_data.get("questions", user_data.get("qa_items", []))

    return {
        "person_name": user_data["person_name"],
        "person_id": user_data["person_id"],
        "context": user_data["context"],
        "qa_items": [convert_question(q) for q in questions],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CloneMemBench data for evaluation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to CloneMemBench data/releases directory")
    parser.add_argument("--context_len", type=str, default="100k", choices=["100k", "500k"],
                        help="Context length subset to use")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"],
                        help="Language to use (en or zh)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # Default data dir is relative to this script
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        args.data_dir = str(script_dir / "data" / "releases")

    data_dir = Path(args.data_dir) / args.context_len
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect all benchmark files for the requested language
    suffix = f"_benchmark_{args.lang}.json"
    json_files = sorted(data_dir.glob(f"*{suffix}"))
    if not json_files:
        raise FileNotFoundError(f"No benchmark files found matching *{suffix} in {data_dir}")

    print(f"Found {len(json_files)} benchmark file(s) in {data_dir}")

    all_users = []
    for jf in json_files:
        print(f"  Loading {jf.name}...")
        with open(jf, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        converted = convert_user(user_data)
        print(f"    -> {converted['person_name']}: {len(converted['context'])} contexts, {len(converted['qa_items'])} questions")
        all_users.append(converted)

    # Default output path
    if args.output_file is None:
        script_dir = Path(__file__).resolve().parent
        args.output_file = str(script_dir / "data" / f"all_users_benchmark_{args.lang}_{args.context_len}.json")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_users, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_users)} user(s) to {output_path}")
    print(f"Total contexts: {sum(len(u['context']) for u in all_users)}")
    print(f"Total questions: {sum(len(u['qa_items']) for u in all_users)}")


if __name__ == "__main__":
    main()
