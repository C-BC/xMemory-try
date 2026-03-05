# xMemory on CloneMemBench

Evaluate xMemory's hierarchical memory system on the [CloneMemBench](https://github.com/AvatarMemory/CloneMemBench) benchmark for long-term memory in AI Clones.

## Quick Start

```bash
# 1. Activate the xMemory conda environment
conda activate xMemory

# 2. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# 3. Run setup (installs extra deps + prepares data)
cd benchmark/clonemembench
bash setup.sh

# 4. Run the full evaluation pipeline
bash run_xmemory_bench.sh
```

## What This Does

The evaluation pipeline:

1. **Data Preparation** (`prepare_data.py`): Converts CloneMemBench's released data format to the eval-compatible schema
2. **xMemory Retrieval** (`eval_xmemory.py`): Processes each user's digital traces through xMemory's episodic -> semantic -> theme pipeline, then retrieves relevant memories for each question
3. **Recall Metrics** (`eval/compute_auto_metrics_for_clonemem.py`): Computes Recall@K metrics
4. **Answer Generation** (`eval/run_generation.py`): Uses retrieved memories to generate answers via LLM
5. **LLM Metrics** (`eval/compute_llm_metrics_for_clonemem.py`): Evaluates answer quality using LLM-as-judge

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | Yes |

### Run Parameters

```bash
# Run with custom parameters
CONTEXT_LEN=500k \
LLM_MODEL=gpt-4o-mini \
EMBEDDING_MODEL=text-embedding-3-small \
SEARCH_STRATEGY=hybrid \
RETRIEVE_K=20 \
bash run_xmemory_bench.sh
```

| Parameter | Default | Options |
|-----------|---------|---------|
| `CONTEXT_LEN` | `100k` | `100k`, `500k` |
| `LLM_MODEL` | `gpt-4o-mini` | Any OpenAI model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding models |
| `SEARCH_STRATEGY` | `hybrid` | `hybrid`, `vector`, `bm25` |
| `RETRIEVE_K` | `20` | Any integer |
| `LANGUAGE` | `en` | `en`, `zh` |
| `GEN_MODEL` | `gpt-4o-mini` | Model for answer generation |

### Running with vLLM (local models)

For using local models via vLLM, modify `eval_xmemory.py` parameters:

```bash
python eval_xmemory.py \
    --in_file data/all_users_benchmark_en_100k.json \
    --output_dir checkpoints/xmemory/ \
    --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --retrieve_k 20
```

## Directory Structure

```
benchmark/clonemembench/
├── README.md                  # This file
├── setup.sh                   # Setup script
├── run_xmemory_bench.sh       # Main evaluation runner
├── requirements.txt           # Extra dependencies
├── prepare_data.py            # Data format converter
├── eval_xmemory.py            # xMemory evaluation script
├── eval_flat.py               # Flat retrieval baseline
├── eval_oracle.py             # Oracle baseline
├── embedding_retriever.py     # Embedding retrieval utils
├── llm_controller.py          # LLM controller utils
├── data/
│   ├── releases/              # CloneMemBench released data
│   │   ├── 100k/              # Short context (~100k tokens)
│   │   └── 500k/              # Long context (>500k tokens)
│   └── all_users_*.json       # Prepared eval data (generated)
├── eval/
│   ├── run_generation.py      # Answer generation
│   ├── compute_auto_metrics_for_clonemem.py
│   ├── compute_llm_metrics_for_clonemem.py
│   ├── compute_average_recall.py
│   └── compute_average_llm.py
└── checkpoints/               # Output results (generated)
```

## Running Individual Steps

```bash
# Step 1: Prepare data only
python prepare_data.py --context_len 100k --lang en

# Step 2: Run xMemory retrieval only
python eval_xmemory.py \
    --in_file data/all_users_benchmark_en_100k.json \
    --output_dir checkpoints/xmemory/ \
    --retrieve_k 20

# Step 3: Compute recall metrics
python eval/compute_auto_metrics_for_clonemem.py \
    --in_file checkpoints/xmemory/xmemory_USER_ID_hybrid.json \
    --qa_file data/all_users_benchmark_en_100k.json

# Step 4: Generate answers
python eval/run_generation.py \
    --input_file data/all_users_benchmark_en_100k.json \
    --retrieval_file checkpoints/xmemory/xmemory_USER_ID_hybrid.json \
    --top_k 20 \
    --model_name gpt-4o-mini

# Step 5: Compute LLM metrics
python eval/compute_llm_metrics_for_clonemem.py \
    --qa_file data/all_users_benchmark_en_100k.json \
    --gen_file checkpoints/xmemory/generation_results/GENERATION_FILE.json \
    --eval_qa
```

## Comparison with Baselines

Run the oracle and flat baselines for comparison:

```bash
# Oracle baseline (upper bound)
python eval_oracle.py \
    --in_file data/all_users_benchmark_en_100k.json \
    --output_dir checkpoints/oracle/

# Flat retrieval baseline
python eval_flat.py \
    --in_file data/all_users_benchmark_en_100k.json \
    --output_dir checkpoints/flat/ \
    --embedding_model stella \
    --retrieve_k 20
```
