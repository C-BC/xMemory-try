# xMemory CloneMemBench Optimization — Project Context

## Goal

Optimize **xMemory** (a hierarchical memory system) to outperform **dense retrieval (Flat)** on the **CloneMemBench** benchmark. Flat indexes each digital trace independently and retrieves via embedding similarity. xMemory uses a multi-layer hierarchy: messages → episodes (LLM-summarized) → semantic memories (atomic facts) → themes.

The hypothesis is that hierarchical memory should enable better long-term memory recall, but it currently underperforms Flat due to information distortion during summarization and granularity mismatches.

## Architecture Overview

### xMemory Pipeline
1. **Messages** arrive as digital traces (social media posts, diary entries, etc.)
2. **Episodes** are created by grouping 2-25 messages and generating LLM summaries
3. **Semantic memories** are atomic facts extracted from episodes
4. **Themes** are higher-level groupings

### Search
- **Hybrid search** (default): BM25 + ChromaDB vector search combined via Reciprocal Rank Fusion (RRF, k=60)
- BM25 results include `original_messages`; ChromaDB results don't (unless enriched)
- `enrich_original_messages=True` was added to trace back from search results to original message content

### Key Design Decision: "Hierarchy for Locating, Originals for Content"
The hierarchy (episodes/semantics) is used to **locate** relevant information, but the actual **content returned** is the original message text, avoiding information distortion from LLM summarization. This is implemented in `eval_xmemory.py`'s `search_and_build_ranked_items()`.

## CloneMemBench Evaluation Pipeline

```
Step 1: eval_xmemory.py        → Retrieval (ranked_items with res_type, chunk_id, content)
Step 2: compute_auto_metrics   → Recall@K (matches chunk_ids against ground truth media_ids)
Step 3: run_generation.py      → Answer generation (LLM generates answers using evidence)
Step 4: compute_llm_metrics    → LLM Judge scores (QA consistency, memory recall, helpfulness)
```

### Key Metrics
- **QA_consistency_score** (0-3): Does the generated answer match the gold answer?
- **Mem_recall** (0-1): Does the retrieved memory cover the ground truth memory?
- **Mem_helpful_score** (0-2): Is the retrieved memory helpful for answering the question?
- All are normalized to 0-1 range for final reporting.

### res_type Compatibility (CRITICAL)
| Script | Filter Logic | What it accepts |
|--------|-------------|-----------------|
| `compute_auto_metrics` | `in ('chunk', 'memory')` | Both (fixed) |
| `run_generation.py` | `in ('chunk', 'memory')` when evidence_type is chunk or memory | Both (fixed) |
| `compute_llm_metrics` (eval_mem='mem') | `!= 'chunk'` | "memory" works |

Our `eval_xmemory.py` outputs `res_type: "memory"`. All three downstream scripts now accept this.

## Files Modified (from original xMemory)

### Benchmark Bridge
- **`benchmark/clonemembench/eval_xmemory.py`** — Main bridge between xMemory and CloneMemBench
  - `search_and_build_ranked_items()`: Uses `enrich_original_messages=True`, extracts per-context content via `context_id_map`
  - `_extract_context_ids_from_messages()`: Gets context_ids from original_messages metadata
  - `_match_context_id()`: Fuzzy matching fallback (Jaccard similarity, threshold 0.15)
  - `res_type` set to `"memory"` (not `"chunk"`)

### Core xMemory Source
- **`src/core/memory_system.py`**
  - Added `enrich_original_messages` param to `search_all()`
  - Added `_enrich_results_with_original_messages()` method
  - For episodes: adds `original_messages` if missing
  - For semantics: traces `source_episodes`/`related_episodes` → episode → `original_messages`, stores as `source_original_messages`

- **`src/api/facade.py`**
  - Added `enrich_original_messages` passthrough to `search()` method

- **`src/search/chroma_search.py`**
  - Extracts `context_id` from `episode.original_messages` metadata during indexing
  - Stores as comma-separated `context_ids` in ChromaDB metadata

### Evaluation Scripts (Downstream)
- **`benchmark/clonemembench/eval/compute_auto_metrics_for_clonemem.py`**
  - Fixed: accepts `res_type in ('chunk', 'memory')` for chunk_id extraction

- **`benchmark/clonemembench/eval/run_generation.py`**
  - Fixed: accepts `res_type in ('chunk', 'memory')` for evidence construction (both in `construct_evidence_text` and `process_item`)

- **`benchmark/clonemembench/eval/compute_llm_metrics_for_clonemem.py`**
  - Uses `!= 'chunk'` filter for eval_mem='mem' — already compatible with "memory"

## Bugs Found and Fixed (Chronological)

1. **Key mismatch "episodes" vs "episodic"**: `eval_xmemory.py` used `results.get("episodes")` but `memory_system.search_all()` returns key `"episodic"`. All episodic results were silently dropped.

2. **res_type "chunk" filtered by LLM judge**: `eval_xmemory.py` set `res_type: "chunk"`, but `compute_llm_metrics` with `eval_mem='mem'` filters `!= 'chunk'`. All memory items were excluded from LLM judge evaluation.

3. **context_id not stored in ChromaDB metadata**: Original code never propagated `context_id` from messages into ChromaDB episode metadata during indexing.

4. **Fuzzy match threshold too high (0.3)**: Lowered to 0.15 for better matching of summarized content to originals.

5. **Information distortion**: Episode summaries and semantic facts lost detail. Fixed by returning original message content instead.

6. **Per-episode content dilution**: All contexts from same episode got identical concatenated content. Fixed by using `context_id_map[ctx_id]` per individual context.

7. **res_type mismatch in downstream scripts** (latest fix): `compute_auto_metrics` and `run_generation.py` filtered for `res_type == 'chunk'` but our items use `"memory"`. This caused recall metrics on empty lists and generation with no evidence.

## Latest Results (Before res_type Fix)

| Metric | xMemory | Flat (Dense) |
|--------|---------|-------------|
| QA_consistency_score_norm | 0.6667 | 0.6875 |
| Mem_helpful_score_norm | 0.6562 | 0.7656 |
| Mem_recall_norm | 0.5653 | 0.7089 |

Note: These results are **unreliable** because Bug #7 (res_type mismatch) meant recall@K was computed on empty item lists and generation had no evidence. Re-running after the fix should give significantly better results.

## How to Run

```bash
cd benchmark/clonemembench

# Full pipeline (will skip retrieval if checkpoints exist due to --resume)
bash run_xmemory_bench.sh --context_len 100k

# To force re-retrieval, delete the checkpoint directory first:
rm -rf checkpoints/xmemory_100k_hybrid/

# Or run steps individually:
python eval_xmemory.py --in_file data/all_users_benchmark_en_100k.json --output_dir checkpoints/xmemory_100k_hybrid/ --retrieve_k 20
python eval/compute_auto_metrics_for_clonemem.py --in_file=<retrieval_file> --qa_file=data/all_users_benchmark_en_100k.json
python eval/run_generation.py --input_file=data/all_users_benchmark_en_100k.json --retrieval_file=<retrieval_file> --top_k=20
python eval/compute_llm_metrics_for_clonemem.py --qa_file=data/all_users_benchmark_en_100k.json --gen_file=<gen_file> --eval_qa --eval_mem=mem
```

## Next Steps / Ideas for Improvement

1. **Re-run benchmark** after res_type fix to get accurate baseline numbers
2. **Embedding re-ranking**: After hierarchy locates candidates, re-rank by embedding similarity between query and original content
3. **Dual-level search**: Search both at episode level AND directly at message level, merge results
4. **Reduce episode size**: Smaller episodes (fewer messages per episode) = less content dilution
5. **Weighted RRF**: Tune the k parameter in RRF fusion, or weight BM25 vs vector differently
6. **Review two arxiv papers**:
   - CloneMemBench paper: https://arxiv.org/pdf/2601.01280
   - Paper questioning graph-based RAG: https://arxiv.org/abs/2601.07023
7. **Consider hybrid approach**: Use xMemory's hierarchy only for complex multi-hop questions, fall back to Flat for simple factual recall

## Project Structure

```
xMemory-try/
├── src/
│   ├── api/facade.py              # xMemory public API
│   ├── core/memory_system.py      # Core memory system
│   ├── search/
│   │   ├── chroma_search.py       # ChromaDB vector search + indexing
│   │   ├── bm25_search.py         # BM25 lexical search
│   │   └── unified_search.py      # Hybrid search (RRF fusion)
│   ├── config.py                  # MemoryConfig
│   └── models.py                  # Data models (Episode, SemanticMemory, etc.)
├── benchmark/clonemembench/
│   ├── eval_xmemory.py            # Bridge: xMemory → CloneMemBench format
│   ├── run_xmemory_bench.sh       # Full pipeline runner
│   ├── eval/
│   │   ├── compute_auto_metrics_for_clonemem.py  # Recall@K
│   │   ├── run_generation.py                      # Answer generation
│   │   └── compute_llm_metrics_for_clonemem.py   # LLM Judge
│   └── data/                      # Benchmark data files
└── CLAUDE.md                      # This file
```

## Environment Notes
- Requires `OPENAI_API_KEY` for LLM calls (GPT-4o-mini default)
- ChromaDB for vector storage
- BM25 for lexical search
- Python with dependencies from requirements.txt
