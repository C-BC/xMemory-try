#!/bin/bash
#
# Run xMemory evaluation on CloneMemBench
#
# This script runs the full pipeline:
#   1. xMemory retrieval (eval_xmemory.py)
#   2. Auto metrics - Recall@K (compute_auto_metrics)
#   3. Answer generation (run_generation.py)
#   4. LLM-based metrics (compute_llm_metrics)
#
# Prerequisites:
#   - Run setup.sh first
#   - Set OPENAI_API_KEY environment variable
#
# Usage:
#   bash run_xmemory_bench.sh [--context_len 100k|500k] [--llm_model MODEL]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Default Parameters ----
CONTEXT_LEN="${CONTEXT_LEN:-100k}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-hybrid}"
RETRIEVE_K="${RETRIEVE_K:-20}"
LANGUAGE="${LANGUAGE:-en}"
GEN_MODEL="${GEN_MODEL:-gpt-4o-mini}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --context_len) CONTEXT_LEN="$2"; shift 2;;
        --llm_model) LLM_MODEL="$2"; shift 2;;
        --embedding_model) EMBEDDING_MODEL="$2"; shift 2;;
        --search_strategy) SEARCH_STRATEGY="$2"; shift 2;;
        --retrieve_k) RETRIEVE_K="$2"; shift 2;;
        --language) LANGUAGE="$2"; shift 2;;
        --gen_model) GEN_MODEL="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

IN_FILE="$SCRIPT_DIR/data/all_users_benchmark_${LANGUAGE}_${CONTEXT_LEN}.json"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints/xmemory_${CONTEXT_LEN}_${SEARCH_STRATEGY}"
OUTFILE_PREFIX="xmemory"

if [ ! -f "$IN_FILE" ]; then
    echo "ERROR: Input file not found: $IN_FILE"
    echo "Please run setup.sh first."
    exit 1
fi

echo "============================================"
echo "  xMemory CloneMemBench Evaluation"
echo "============================================"
echo "Context length: $CONTEXT_LEN"
echo "LLM model: $LLM_MODEL"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Search strategy: $SEARCH_STRATEGY"
echo "Retrieve K: $RETRIEVE_K"
echo "Input: $IN_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================"
echo ""

cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$SCRIPT_DIR:$PYTHONPATH"

# ---- Step 1: Run xMemory Retrieval ----
echo "======== Step 1: xMemory Retrieval ========"
python eval_xmemory.py \
    --in_file "$IN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --outfile_prefix "$OUTFILE_PREFIX" \
    --llm_model "$LLM_MODEL" \
    --embedding_model "$EMBEDDING_MODEL" \
    --search_strategy "$SEARCH_STRATEGY" \
    --retrieve_k "$RETRIEVE_K" \
    --language "$LANGUAGE" \
    --resume

echo ""
echo "======== Step 2: Compute Retrieval Metrics (Recall@K) ========"
# Compute auto metrics for each retrieval result file
for RETRIEVAL_FILE in "$OUTPUT_DIR"/${OUTFILE_PREFIX}_*.json; do
    [ -f "$RETRIEVAL_FILE" ] || continue
    echo "  Computing metrics for: $(basename $RETRIEVAL_FILE)"
    python eval/compute_auto_metrics_for_clonemem.py \
        --in_file="$RETRIEVAL_FILE" \
        --qa_file="$IN_FILE" \
        2>&1 || echo "  Warning: metrics computation failed for $(basename $RETRIEVAL_FILE)"
done

echo ""
echo "======== Step 3: Generate Answers (top_k=5, 10, 20) ========"
for TOP_K in 5 10 20; do
    for RETRIEVAL_FILE in "$OUTPUT_DIR"/${OUTFILE_PREFIX}_*.json; do
        [ -f "$RETRIEVAL_FILE" ] || continue
        echo "  Generating answers (top_k=$TOP_K) for: $(basename $RETRIEVAL_FILE)"
        python eval/run_generation.py \
            --input_file="$IN_FILE" \
            --retrieval_file="$RETRIEVAL_FILE" \
            --top_k=$TOP_K \
            --model_name="$GEN_MODEL" \
            2>&1 || echo "  Warning: generation failed for $(basename $RETRIEVAL_FILE) top_k=$TOP_K"
    done
done

echo ""
echo "======== Step 4: Compute LLM Metrics ========"
GEN_DIR="$OUTPUT_DIR/generation_results"
if [ -d "$GEN_DIR" ]; then
    for GEN_FILE in "$GEN_DIR"/*.json; do
        [ -f "$GEN_FILE" ] || continue
        # Only compute LLM metrics for top20 by default
        if [[ "$GEN_FILE" == *"top20"* ]]; then
            echo "  Computing LLM metrics for: $(basename $GEN_FILE)"
            python eval/compute_llm_metrics_for_clonemem.py \
                --qa_file="$IN_FILE" \
                --gen_file="$GEN_FILE" \
                --eval_qa \
                --eval_mem="mem" \
                2>&1 || echo "  Warning: LLM metrics failed for $(basename $GEN_FILE)"
        fi
    done
else
    echo "  No generation results found in $GEN_DIR"
fi

echo ""
echo "============================================"
echo "  Evaluation Complete!"
echo ""
echo "  Results directory: $OUTPUT_DIR"
echo "  Retrieval metrics: $OUTPUT_DIR/recall_metrics/"
echo "  Generation results: $OUTPUT_DIR/generation_results/"
echo "============================================"
