#!/bin/bash
#
# Setup script for running xMemory on CloneMemBench
#
# Prerequisites:
#   - xMemory conda environment already activated (conda activate xMemory)
#   - OPENAI_API_KEY set in environment
#
# Usage:
#   bash setup.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "  xMemory CloneMemBench Setup"
echo "============================================"
echo ""
echo "Repo root: $REPO_ROOT"
echo "Benchmark dir: $SCRIPT_DIR"
echo ""

# ---- 1. Install additional CloneMemBench dependencies ----
echo "[1/3] Installing additional dependencies for CloneMemBench evaluation..."

pip install --quiet \
    json_repair>=0.25.0 \
    backoff>=2.2.1 \
    tiktoken>=0.9.0 \
    2>&1 | tail -3

# Verify key packages
python -c "
import json_repair, backoff, tiktoken, openai, torch, tqdm, pydantic
print('  All required packages verified.')
"

# ---- 2. Prepare benchmark data ----
echo ""
echo "[2/3] Preparing benchmark data..."

cd "$SCRIPT_DIR"

# Prepare 100k English data
if [ ! -f "data/all_users_benchmark_en_100k.json" ]; then
    echo "  Converting 100k English benchmark data..."
    python prepare_data.py --context_len 100k --lang en
else
    echo "  100k English data already prepared."
fi

# Prepare 500k English data
if [ ! -f "data/all_users_benchmark_en_500k.json" ]; then
    echo "  Converting 500k English benchmark data..."
    python prepare_data.py --context_len 500k --lang en
else
    echo "  500k English data already prepared."
fi

# ---- 3. Verify setup ----
echo ""
echo "[3/3] Verifying xMemory import..."

cd "$REPO_ROOT"
python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from xMemory import xMemory, MemoryConfig
print('  xMemory import successful.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To run the benchmark:"
echo "    bash run_xmemory_bench.sh"
echo ""
echo "  Or run manually:"
echo "    python eval_xmemory.py \\"
echo "      --in_file data/all_users_benchmark_en_100k.json \\"
echo "      --output_dir checkpoints/xmemory/ \\"
echo "      --llm_model gpt-4o-mini \\"
echo "      --retrieve_k 20"
echo "============================================"
