#!/usr/bin/env python3
"""
Supported embedding models:
- SentenceTransformer: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
- Contriever: facebook/contriever
- Stella: dunzhang/stella_en_1.5B_v5
- GTE: Alibaba-NLP/gte-Qwen2-7B-instruct

"""

import sys
import os
import json
import json_repair
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import multiprocessing as mp
from functools import partial
import torch
from pydantic import BaseModel, Field
from typing import List
import backoff
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from embedding_retriever import FlexibleEmbeddingRetriever
from llm_controller import LLMController


class QueryResponse(BaseModel):
    keywords: str = Field(..., description="Generated keywords for the query")

# Model name mapping
MODEL_CONFIG = {
    'all-MiniLM-L6-v2': {'model_name': 'all-MiniLM-L6-v2', 'model_type': 'sentence-transformer'},
    'all-mpnet-base-v2': {'model_name': 'all-mpnet-base-v2', 'model_type': 'sentence-transformer'},
    'contriever': {'model_name': 'facebook/contriever', 'model_type': 'contriever'},
    'stella': {'model_name': 'dunzhang_stella_en_1.5B_v5', 'model_type': 'stella'},
    'gte': {'model_name': 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'model_type': 'gte'},
    'bm25': {'model_name': 'bm25', 'model_type': 'bm25'},  # BM25 model (no embedding)
    'openai': {'model_name': 'text-embedding-3-small', 'model_type': 'openai'}  # OpenAI embedding
}

@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

class UserFacts(BaseModel):
    """Pydantic model for structured output of user facts."""
    facts: List[str]


class SubQuestions(BaseModel):
    """Pydantic model for structured output of question decomposition."""
    subquestions: List[str] = Field(..., description="List of decomposed subquestions")


class FlexibleAmemRetriever:
    """A-mem retriever with flexible embedding model support"""
    
    def __init__(self, 
                 embedding_model='all-MiniLM-L6-v2',
                 llm_model='gpt-4o-mini', 
                 backend='openai', 
                 retrieve_k=10,
                 device=None,
                 cache_dir=None,
                 api_key=None,
                 vllm_base_url=None):
        
        # Get model config
        if embedding_model in MODEL_CONFIG:
            model_config = MODEL_CONFIG[embedding_model]
        else:
            # Assume it's a sentence-transformer model
            model_config = {'model_name': embedding_model, 'model_type': 'sentence-transformer'}
        
        self.model_config = model_config
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.device = device
        
        self.retriever_llm = LLMController(backend=backend, model=llm_model, api_key=api_key, base_url=vllm_base_url)
        self.retrieve_k = retrieve_k if retrieve_k > 0 else 100000  # get all memories
        
        self.llm_model = llm_model
        self.client = OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY') if 'gpt' in llm_model else 'EMPTY',
            base_url=None if 'gpt' in llm_model else vllm_base_url,
        )
        
        # Track corpus IDs for mapping retrieved memories back
        self.corpus_ids = []
        self.corpus_content = []
        self.corpus_timestamps = []

        # track expansions
        self.expansions = []

    def _decompose_question(self, question, num_subquestions=3):
        """
        Decompose a question into multiple subquestions using LLM.
        
        Args:
            question: The original question to decompose
            num_subquestions: Number of subquestions to generate
            
        Returns:
            List of subquestions
        """
        decomposition_prompt = f"""Given the following question, decompose it into maximum {num_subquestions} more specific subquestions that would help answer the original question. Each subquestion should focus on a different aspect or detail of the original question.

Original Question: {question}

Generate up to {num_subquestions} subquestions that are:
1. More specific and focused than the original
2. Cover different aspects of the original question
3. Can be answered independently

Provide the subquestions as a list."""

        kwargs = {
            'model': self.llm_model,
            'messages': [
                {"role": "user", "content": decomposition_prompt}
            ],
            'n': 1,
            'temperature': 0.7,  # Slightly higher temperature for more diverse subquestions
            'max_tokens': 500
        }
        
        if 'gpt' in self.llm_model:
            kwargs['response_format'] = SubQuestions
            completion = self.client.chat.completions.parse(**kwargs)
        else:
            kwargs['extra_body'] = {'guided_json': SubQuestions.model_json_schema()}
            completion = self.client.chat.completions.create(**kwargs)

        subquestions_text = completion.choices[0].message.content
        subquestions_text = json_repair.loads(subquestions_text)
        
        parsed_response = SubQuestions(**subquestions_text)
        if parsed_response is not None and parsed_response.subquestions:
            return parsed_response.subquestions[:num_subquestions]
        
        # Fallback: return original question if decomposition fails
        return [question]

    def _extract_summary(self, content):
        summarization_prompt = "Below are some personal logs and conversations. Please summarize them as concisely as possible in a short paragraph, extracting the main themes and key information. Content:\n"
        summarization_prompt += content
        summarization_prompt += "\n\nSummary (be concise):"
        kwargs = {
            'model': self.llm_model,
            'messages':[
                {"role": "user", "content": summarization_prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 500
        }
        completion = chat_completions_with_backoff(self.client, **kwargs)
        return completion.choices[0].message.content.strip()
    
    def _extract_keywords(self, content):
        summarization_prompt = "Below are some personal logs and conversations. Generate a list of keyphrases for the session. Separate each keyphrase with a semicolon. Content:\n"
        summarization_prompt += content
        summarization_prompt += "\n\nKeyphrases (separated by semicolon):"
        kwargs = {
            'model': self.llm_model,
            'messages':[
                {"role": "user", "content": summarization_prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 200,
        }
        completion = chat_completions_with_backoff(self.client, **kwargs)
        return completion.choices[0].message.content.strip()
    
    def _extract_facts(self, content):
        summarization_prompt = "Below are some personal logs and conversations. Extract all factual statements, personal information, life events, experience, and preferences. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list. Content:\n"
        summarization_prompt += content
        summarization_prompt += "\n\nExtracted Facts (one per line):"
        kwargs = {
            'model': self.llm_model,
            'messages':[
                {"role": "user", "content": summarization_prompt}
            ],
            'n': 1,
            'temperature': 0,
            'max_tokens': 2000,
        }
        
        if 'gpt' in self.llm_model:
            kwargs['response_format'] = UserFacts
            completion = self.client.chat.completions.parse(**kwargs)
        else:
            kwargs['extra_body'] = {'guided_json': UserFacts.model_json_schema()}
            completion = self.client.chat.completions.create(**kwargs)

        facts_text = completion.choices[0].message.content
        facts_text = json_repair.loads(facts_text)

        parsed_response = UserFacts(**facts_text)
        if parsed_response is not None:
            return parsed_response.facts
            
        return None

    def _extract_expansion(self, content, index_expansion_methods, max_retries=10):
        """
        Extract expansions with retry mechanism for each method.
        
        Args:
            content: The content to extract expansions from
            index_expansion_methods: List of expansion methods to use
            max_retries: Maximum number of retries for each extraction method
            
        Returns:
            Dict mapping method name to extracted expansion (or empty string if all retries fail)
        """
        expansions = {}
        
        for method in index_expansion_methods:
            success = False
            result = None
            
            for attempt in range(max_retries):
                try:
                    if method == 'summ':
                        result = self._extract_summary(content)
                        success = True
                        break
                    elif method == 'keywords':
                        result = self._extract_keywords(content)
                        success = True
                        break
                    elif method == 'facts':
                        facts_list = self._extract_facts(content)
                        result = '; '.join(facts_list) if facts_list is not None else ''
                        success = True
                        break
                    else:
                        print(f"⚠️  Unknown expansion method: {method}")
                        break
                        
                except Exception as e:
                    print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed for method '{method}': {str(e)}")
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        raise Exception(f"❌ All retries exhausted for method '{method}'. Using empty string as fallback.")
            
            # Store result (either successful extraction or empty string after all retries)
            expansions[method] = result if result is not None else ''
                
        return expansions
    
    
    def add_memory(self, content, time_stamp, corpus_id, args, expansions=None):
        """Add memory and track its corpus ID with retry logic
        
        Args:
            content: The content of the memory
            time_stamp: Timestamp of the memory
            corpus_id: Original corpus ID for tracking
            keywords: Pre-extracted keywords (if cached, skips LLM extraction)
            summary: Pre-extracted summary (if cached, skips LLM extraction)
            facts: Pre-extracted user facts (if cached, skips LLM extraction)
        """
        self.corpus_content.append(content)
        self.corpus_timestamps.append(time_stamp)
        self.corpus_ids.append(corpus_id)
        
        if args.index_expansion_method:
            if expansions is None:  # No cached expansions, extract now
                expansions = self._extract_expansion(content, args.index_expansion_method)
            self.expansions.append(expansions)        

    def retrieve(self, question, use_expansion, expansion_join_method, use_chunk):
        """
        First initialize retriever embeddings after finishing adding memories and before do retrieval, 
        different from A-Mem or Mem0 since there is no memory update.
        
        Follows the expansion join logic from index_expansion_utils.py:
        - Build corpus with original content
        - Add expansions for each item (separate items)
        - If 'merge' in join_method, merge all items with same corpus_id at the end
        
        Returns: list of indices into self.corpus_ids
        """

        # Step 1: Build corpus with original content and separate expansions
        corpus = []
        corpus_ids = []
        corpus_type = []  # Track whether each item is 'original' or an expansion type
        corpus_timestamps = []
        
        for i, (content, corpus_id, timestamp) in enumerate(zip(self.corpus_content, self.corpus_ids, self.corpus_timestamps)):
            # Always add original content first (similar to resolve_expansion adding to existing_corpus)
            corpus.append(content)
            corpus_ids.append(corpus_id)
            corpus_type.append('original')
            corpus_timestamps.append(timestamp)
            
            # Add expansions if requested
            if use_expansion and expansion_join_method != 'none':
                for method in use_expansion:
                    # if method in self.expansions[i] and self.expansions[i][method]:
                    expansion_text = self.expansions[i][method]
                    # Normalize: trim whitespace and skip empty
                    if expansion_text and expansion_text.strip():
                        corpus.append(expansion_text.strip())
                        corpus_ids.append(corpus_id)
                        corpus_type.append(method)  # Track the expansion type (e.g., 'summ', 'keywords', 'facts')
                        corpus_timestamps.append(timestamp)
        
        # Step 2: Apply merge strategy if needed (similar to resolve_multiple_expansions)
        if 'merge' in expansion_join_method and use_expansion:
            # Group items by corpus_id and merge them
            merged_corpus = []
            merged_corpus_ids = []
            merged_corpus_type = []
            merged_corpus_timestamps = []
            
            # Track which corpus_ids we've already processed
            processed_ids = set()
            
            for cid in self.corpus_ids:  # Iterate in original order
                if cid in processed_ids:
                    continue
                processed_ids.add(cid)
                
                if expansion_join_method == 'merge_raw':  # also merge original content into expansion
                    indices = [idx for idx, c_id in enumerate(corpus_ids) if c_id == cid]
                else:
                    indices = [idx for idx, (c_id, c_type) in enumerate(zip(corpus_ids, corpus_type)) 
                                if c_id == cid and c_type != 'original']
                
                # Add the expansions first
                # Merge the items
                merged_text = '\n'.join([corpus[idx] for idx in indices])
                merged_corpus.append(merged_text)
                merged_corpus_ids.append(cid)

                # Determine merged type label
                types_in_merge = [corpus_type[idx] for idx in indices]
                if expansion_join_method == 'merge_raw':
                    expansion_types = [t for t in types_in_merge]
                else:
                    expansion_types = [t for t in types_in_merge if t != 'original']
                
                if expansion_types:
                    merged_corpus_type.append(f'merged-{"_".join(sorted(set(expansion_types)))}')
                else:
                    merged_corpus_type.append('merged')
                
                merged_corpus_timestamps.append(corpus_timestamps[indices[0]])
                
                # For merge, also need to keep the original when use_chunk is True
                if expansion_join_method == 'merge' and use_chunk:
                    original_idx = [idx for idx, (c_id, c_type) in enumerate(zip(corpus_ids, corpus_type)) 
                                   if c_id == cid and c_type == 'original']
                    if original_idx:
                        merged_corpus.append(corpus[original_idx[0]])
                        merged_corpus_ids.append(cid)
                        merged_corpus_type.append('original')
                        merged_corpus_timestamps.append(corpus_timestamps[original_idx[0]])
            
            # Replace corpus with merged version
            corpus = merged_corpus
            corpus_ids = merged_corpus_ids
            corpus_type = merged_corpus_type
            corpus_timestamps = merged_corpus_timestamps
        
        elif expansion_join_method == 'separate':
            # For 'separate': keep items separate but optionally remove original if not use_chunk
            if not use_chunk:
                # Remove original items
                filtered_corpus = []
                filtered_ids = []
                filtered_types = []
                filtered_timestamps = []
                
                for c, cid, ctype, cts in zip(corpus, corpus_ids, corpus_type, corpus_timestamps):
                    if ctype != 'original':
                        filtered_corpus.append(c)
                        filtered_ids.append(cid)
                        filtered_types.append(ctype)
                        filtered_timestamps.append(cts)
                
                corpus = filtered_corpus
                corpus_ids = filtered_ids
                corpus_type = filtered_types
                corpus_timestamps = filtered_timestamps
        
        # Step 3: Initialize retriever and add documents
        retriever = FlexibleEmbeddingRetriever(
            model_name=self.model_config['model_name'],
            model_type=self.model_config['model_type'],
            device=self.device,
            cache_dir=self.cache_dir,
            api_key=self.api_key,
        )
        
        # Add all documents to retriever
        retriever.add_documents(corpus)
        
        # Step 4: Perform retrieval and score aggregation
        # Similar to expand_run_retrieval.py's scoring logic
        
        # Get raw scores for all documents
        if self.model_config['model_type'] == 'bm25':
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(question.split(" "))
        else:
            # For dense retrievers, get similarity scores
            query_embedding = retriever.encode([question])
            corpus_embeddings = retriever.embeddings
            
            similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
            scores = similarities
        
        # Group scores by corpus_id and type (following expand_run_retrieval.py logic)
        scores_by_id = {cid: {} for cid in np.unique(corpus_ids)}
        for i in range(len(corpus_ids)):
            cid, ctype = corpus_ids[i], corpus_type[i]
            if ctype not in scores_by_id[cid]:
                scores_by_id[cid][ctype] = [scores[i]]
            else:
                scores_by_id[cid][ctype] += [scores[i]]
        
        # Average scores for each type for each corpus_id
        for cid in scores_by_id:
            for ctype in list(scores_by_id[cid].keys()):
                if not scores_by_id[cid][ctype]:
                    del scores_by_id[cid][ctype]
                    continue
                scores_by_id[cid][ctype] = np.mean(scores_by_id[cid][ctype])
            
            # Compute final score based on join method and use_chunk setting
            # Following expand_run_retrieval.py: use_raw_session_as_key controls whether to use 'original'
            # For our case:
            # - use_chunk=True: use all available scores
            # - use_chunk=False with merge_raw: ERROR - shouldn't happen (merge_raw always includes original)
            # - use_chunk=False with merge/separate: use only expansion scores

            if use_chunk:  # use all available scores
                scores_by_id[cid]['final'] = np.mean(list(scores_by_id[cid].values()))
            else:  # only use expansion scores (exclude original)
                # For merge_raw, the merged item contains original, so this shouldn't be used
                if expansion_join_method == 'merge_raw':
                    # merge_raw always includes original in the merge, so use_chunk should be True
                    # If somehow use_chunk=False, we should raise an error or fall back to using all scores
                    raise Exception(f"merge_raw with use_chunk=False is contradictory. Using all scores.")
                else:
                    # For merge or separate, exclude 'original' type scores
                    expansion_scores = [v for k, v in scores_by_id[cid].items() if k != 'original']
                    if not expansion_scores:
                        # This can happen if there are no expansions for this corpus_id
                        raise Exception(f"WARNING: No expansion scores found for corpus_id {cid} when not using chunk. Using original score as fallback.")
                    else:
                        scores_by_id[cid]['final'] = np.mean(expansion_scores)
        
        # Rank corpus_ids based on final scores
        ranked_corpus_ids = sorted(scores_by_id.items(), key=lambda x: x[1]['final'], reverse=True)
        ranked_corpus_ids = [cid for cid, _ in ranked_corpus_ids[:self.retrieve_k]]
        
        # Map back to original indices in self.corpus_ids
        result_indices = []
        for cid in ranked_corpus_ids:
            idx = self.corpus_ids.index(cid)
            result_indices.append(idx)
        
        return result_indices


def load_per_context_cache(context_id, per_context_cache_dir):
    """Load cached metadata for a specific context item"""
    cache_file = os.path.join(per_context_cache_dir, f"{context_id}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load per-context cache for {context_id}: {str(e)}")
    return None


def save_per_context_cache(context_id, metadata, per_context_cache_dir):
    """Save metadata for a specific context item"""
    os.makedirs(per_context_cache_dir, exist_ok=True)
    cache_file = os.path.join(per_context_cache_dir, f"{context_id}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to save per-context cache for {context_id}: {str(e)}")


def process_user(user_data, save_path, args, device=None):
    """
    Process user data with temporal awareness: for each question, only retrieve from 
    contexts that occurred before the question time.
    
    Uses two-tier caching:
    1. Per-context cache: Stores extracted metadata (keywords, context, tags) for each context item
    2. Per-question final cache: Stores complete memory system state at the time of each question
    """
    user_name = user_data["person_name"]
    user_id = user_data["person_id"]
    sessions = user_data["context"]

    # Use provided device or auto-detect
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate cache paths
    # NOTE: Different LLM models will extract different metadata, so cache separately
    llm_model_name = args.llm_model.replace('/', '_')  # Replace / in model names for filename
    
    # Per-context cache directory (individual metadata for each context)
    per_context_cache_dir = os.path.join(args.memories_dir, f"{user_id}_{llm_model_name}_per_context")    
    os.makedirs(args.memories_dir, exist_ok=True)

    # Initialize retriever
    retriever = FlexibleAmemRetriever(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            device=device,
            cache_dir=args.cache_dir,
            api_key=args.api_key,
            vllm_base_url=args.vllm_base_url
        )

    # Sort contexts by event_date for temporal processing
    contexts = sorted(user_data['context'], key=lambda x: datetime.fromisoformat(x["event_date"]))
    
    # Sort questions by question_time
    qa_items = sorted(user_data['qa_items'], key=lambda x: datetime.fromisoformat(x['question_time']))
    
    print(f"🔍 Processing {len(contexts)} context items and {len(qa_items)} questions with temporal awareness...")
    
    # Check if per-context cache is available
    per_context_cache_available = os.path.exists(per_context_cache_dir)
    if per_context_cache_available:
        cached_count = len([f for f in os.listdir(per_context_cache_dir) if f.endswith('.json')])
        print(f"Found per-context cache directory with {cached_count} cached items")
    
    # Track which contexts have been added
    added_context_ids = set()
    context_pointer = 0  # Points to the next context to add
    
    results = {}
    
    # Process each question in temporal order
    for question_dict in tqdm(qa_items, desc=f"Processing questions for {user_name}"):
        question_id = question_dict['id']
        question = question_dict['question']
        question_time_str = question_dict['question_time']
        question_time = datetime.fromisoformat(question_time_str)
        
        # Add all contexts that occurred before this question and haven't been added yet
        cache_hits = 0
        cache_misses = 0
        
        while context_pointer < len(contexts):
            ctx = contexts[context_pointer]
            ctx_time = datetime.fromisoformat(ctx["event_date"])
            
            # Stop if this context is after the question time
            if ctx_time > question_time:
                break
            
            context_id = ctx['id']
            
            # Skip if already added
            if context_id in added_context_ids:
                context_pointer += 1
                continue
            
            timestamp = str(ctx_time)
            
            # Try to load per-context cache
            cached_metadata = None
            if args.index_expansion_method:
                cached_metadata = load_per_context_cache(context_id, per_context_cache_dir)

                # Check if cached metadata has all required expansion methods
                if cached_metadata is not None:
                    # Filter to only use expansion methods that are both cached and requested
                    available_methods = set(cached_metadata.keys())
                    requested_methods = set(args.index_expansion_method)
                    
                    # Check if we have all requested methods in cache
                    if not requested_methods.issubset(available_methods):
                        missing_methods = requested_methods - available_methods
                        print(f"⚠️  Cache for {context_id} is missing methods: {missing_methods}. Re-extracting.")
                        cached_metadata = None  # Force re-extraction
                    else:
                        # Filter cached metadata to only include requested methods
                        cached_metadata = {k: v for k, v in cached_metadata.items() if k in args.index_expansion_method}
                
                
            if cached_metadata is not None and not args.force_reextract:
                # Use cached metadata - pass to add_memory to skip LLM extraction
                retriever.add_memory(
                    content=ctx['content'],
                    time_stamp=timestamp,
                    corpus_id=context_id,
                    args=args,
                    expansions=cached_metadata
                )
                cache_hits += 1
            else:
                # No cache available, extract metadata using LLM
                retriever.add_memory(
                    content=ctx['content'],
                    time_stamp=timestamp,
                    corpus_id=context_id,
                    args=args
                )
                
                # Save per-context cache for future use
                if args.index_expansion_method:
                    metadata_to_cache = retriever.expansions[-1]
                    save_per_context_cache(context_id, metadata_to_cache, per_context_cache_dir)
                    cache_misses += 1
            
            added_context_ids.add(context_id)
            context_pointer += 1
        
        if cache_hits > 0 or cache_misses > 0:
            print(f"  Question {question_id}: Added {cache_hits + cache_misses} contexts ({cache_hits} cached, {cache_misses} new)")
        
        # Handle question decomposition if enabled
        if args.num_subquestions > 0:
            # Decompose question into subquestions
            try:
                subquestions = retriever._decompose_question(question, num_subquestions=args.num_subquestions)
                print(f"  Decomposed into {len(subquestions)} subquestions")
            except Exception as e:
                print(f"⚠️  Failed to decompose question {question_id}: {str(e)}")
                subquestions = [question]  # Fallback to original question
        else:
            # No decomposition, use original question
            subquestions = [question]
        
        # Perform retrieval for each subquestion
        all_ranked_items = []
        for subq_idx, subquestion in enumerate(subquestions):
            memory_indices = retriever.retrieve(
                question=subquestion,
                use_expansion=args.index_expansion_method,
                expansion_join_method=args.join_method,
                use_chunk=args.use_chunk
            )
            
            # Build ranked items for this subquestion
            for rank, idx in enumerate(memory_indices):
                corpus_id = retriever.corpus_ids[idx]
                raw_content = retriever.corpus_content[idx]
                timestamp = retriever.corpus_timestamps[idx]

                ranked_item = {
                    'res_type': 'chunk', 
                    'rank': rank, 
                    'chunk_id': corpus_id, 
                    'content': raw_content, 
                    'timestamp': timestamp,
                    'subquestion': subquestion,
                    'subquestion_id': subq_idx
                }
                all_ranked_items.append(ranked_item)

                if args.index_expansion_method:
                    for method in args.index_expansion_method:
                        expansion_content = retriever.expansions[idx].get(method, '')
                        if expansion_content:
                            expansion_item = {
                                'res_type': f'extract_{method}', 
                                'rank': rank, 
                                'chunk_id': corpus_id, 
                                'content': expansion_content, 
                                'timestamp': timestamp,
                                'subquestion': subquestion,
                                'subquestion_id': subq_idx
                            }
                            all_ranked_items.append(expansion_item)

        results[question_id] = {
            'question': question, 
            'question_time': question_time_str,
            'num_contexts_processed': len(added_context_ids),
            'subquestions': subquestions if args.num_subquestions > 0 else None,
            'ranked_items': all_ranked_items
        }

    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved results for {user_name} to {save_path}")
    print(f"📊 Final statistics: {len(added_context_ids)} contexts processed, {len(results)} questions answered")
    return results


def process_user_wrapper(user_data_with_args):
    """
    Wrapper function for multiprocessing.
    Unpacks user_data and args, then calls process_user.
    Supports GPU assignment for parallel processing.
    """
    user_data, args, gpu_id, output_dir, timestamp = user_data_with_args
    user_name = user_data.get('person_name', 'unknown')
    user_id = user_data.get('person_id', 'unknown')
    
    # Set device based on GPU ID
    if gpu_id is not None and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
        print(f"🎯 Process assigned to GPU {gpu_id} for user {user_name}")
    else:
        device = 'cpu'
        print(f"🎯 Process assigned to CPU for user {user_name}")
    
    # Generate output filename
    subq_suffix = f"_subq{args.num_subquestions}" if args.num_subquestions > 0 else ""
    output_filename = f"{args.outfile_prefix}_{user_id}_{args.embedding_model}-{'_'.join(args.index_expansion_method) if args.index_expansion_method else 'no_expansion'}-{args.join_method}-{'use_chunk' if args.use_chunk else 'no_chunk'}{subq_suffix}.json"
    save_path = os.path.join(output_dir, output_filename)
    
    # Check if already processed (resume functionality)
    if args.resume and os.path.exists(save_path):
        print(f"⏭️  Skipping {user_name} - already processed at {save_path}")
        with open(save_path, 'r', encoding='utf-8') as f:
            user_results = json.load(f)
        return (user_id, user_results, save_path, None)
    
    print(f"\n{'='*60}")
    print(f"Processing user: {user_name} ({user_id}) on {device}")
    print(f"Context items: {len(user_data.get('context', []))}")
    print(f"QA items: {len(user_data.get('qa_items', []))}")
    print(f"{'='*60}\n")
    
    try:
        user_results = process_user(user_data, save_path, args, device=device)
        return (user_id, user_results, save_path, None)
    except Exception as e:
        error_msg = f"❌ Error processing {user_name} on {device}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return (user_id, None, save_path, error_msg)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any CUDA operations
    mp.set_start_method('spawn', force=True)
    
    def list_of_str(value):
        """Custom argparse type to parse a comma-separated list of strings"""
        return value.split(',')

    parser = argparse.ArgumentParser(description="A-mem Retrieval Stage with Flexible Model Support (Stage 1)")
    parser.add_argument('--in_file', type=str, default='data/new_data/all_users_benchmark_en.json', help='Input JSON file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/new_data/flatten/', help='Output directory')
    parser.add_argument('--outfile_prefix', type=str, default='default', help='Output file prefix')

    parser.add_argument('--index_expansion_method', type=list_of_str, default=None,
                        help='Index expansion methods to use (comma-separated), e.g., "summ,keywords,facts"')
    parser.add_argument('--join_method', type=str, default='none', choices=['none', 'merge', 'merge_raw', 'separate'])
    parser.add_argument('--use_chunk', action='store_true', help='Use original chunk for retrieval when adding expansion; must be true if merge_raw', default=False)

    # Model configuration
    parser.add_argument('--embedding_model', type=str, default='contriever',
                       choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'contriever', 'stella', 'gte', 'openai', 'bm25'],
                       help='Embedding model for retrieval')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='LLM model for query expansion')
    parser.add_argument('--backend', type=str, default='openai', choices=['openai', 'ollama'], help='LLM backend')
    parser.add_argument('--retrieve_k', type=int, default=50, help='Number of memories to retrieve')

    # Question decomposition
    parser.add_argument('--num_subquestions', type=int, default=0, help='Number of subquestions to decompose each question into (0 = no decomposition)')

    parser.add_argument('--api_key', type=str, default='', help='API key for LLM backend (if needed)')
    parser.add_argument('--vllm_base_url', type=str, default='http://localhost:8001/v1', help='Base URL for vLLM server (if using self-hosted)')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default=None, help='Device for embedding model (e.g., cuda, cuda:0, cpu). If not specified, auto-detects GPUs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers. With GPUs, workers are distributed across available GPUs in round-robin fashion (default: 1)')
    parser.add_argument('--cache_dir', type=str, default='/data/users1/ywei/data/cache/', help='Cache directory for models')
    
    # Cache configuration
    parser.add_argument('--force_reextract', action='store_true', help='Force re-extraction of memories even if cache exists', default=False)
    
    # Resume configuration
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoints and skip already-processed samples', default=True)
    
    args = parser.parse_args()

    if args.vllm_base_url == 'none':
        args.vllm_base_url = None

    args.use_chunk = True if args.join_method == 'merge_raw' else args.use_chunk
    if args.index_expansion_method is not None: assert args.join_method != 'none', "Index expansion methods require a join method other than 'none'"
    
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
    args.memories_dir = os.path.join(output_dir, 'memories')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.memories_dir, exist_ok=True)
    
    # Determine device(s) and GPU availability
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.device:
        # User specified device explicitly
        device_mode = args.device
        use_gpus = args.device.startswith('cuda')
        print(f"Using user-specified device: {device_mode}")
    elif num_gpus > 0:
        # GPUs available - use them
        device_mode = 'cuda'
        use_gpus = True
        print(f"🎮 Detected {num_gpus} GPU(s)")
    else:
        # No GPUs available - use CPU
        device_mode = 'cpu'
        use_gpus = False
        print(f"No GPUs detected, using CPU")
    
    # Determine number of workers
    if args.num_workers > 1:
        if use_gpus and num_gpus > 0:
            # Limit workers to number of GPUs if not specified
            max_workers = min(args.num_workers, num_gpus)
            if args.num_workers > num_gpus:
                print(f"⚠️  Requested {args.num_workers} workers but only {num_gpus} GPU(s) available.")
                print(f"   Setting workers to {max_workers} to match GPU count.")
            num_workers = max_workers
        else:
            num_workers = args.num_workers
    else:
        num_workers = 1
    
    print(f"Embedding model: {args.embedding_model}")
    print(f"LLM model: {args.llm_model}")
    print(f"Index expansion methods: {args.index_expansion_method}")
    print(f"Join method: {args.join_method}")
    print(f"Number of subquestions: {args.num_subquestions if args.num_subquestions > 0 else 'disabled (no decomposition)'}")
    print(f"Number of workers: {num_workers}")
    
    # Generate a single timestamp for all users to maintain consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process each user
    all_results = {}
    
    if num_workers > 1:
        # Multiprocessing mode with GPU assignment
        print(f"\n🚀 Using multiprocessing with {num_workers} workers")
        
        if use_gpus and num_gpus > 0:
            # Assign users to GPUs in round-robin fashion
            print(f"📍 Distributing {len(users_data)} users across {num_gpus} GPU(s)")
            user_args = []
            for idx, user_data in enumerate(users_data):
                gpu_id = idx % num_gpus  # Round-robin assignment
                user_args.append((user_data, args, gpu_id, output_dir, timestamp))
            
            # Show GPU assignment distribution
            gpu_counts = {}
            for _, _, gpu_id, _, _ in user_args:
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
            print(f"   GPU assignment distribution: {gpu_counts}")
        else:
            # CPU-only mode
            print(f"📍 Processing on CPU")
            user_args = [
                (user_data, args, -1, output_dir, timestamp)  # -1 indicates CPU
                for user_data in users_data
            ]
        
        # Process users in parallel
        with mp.Pool(processes=num_workers) as pool:
            results_list = pool.map(process_user_wrapper, user_args)
        
        # Collect results
        for user_id, user_results, save_path, error_msg in results_list:
            if user_results is not None:
                all_results[user_id] = user_results
            if error_msg:
                print(error_msg)
    else:
        # Sequential processing (original behavior)
        print(f"\n📝 Sequential processing (single worker)")
        gpu_id = 0 if use_gpus and num_gpus > 0 else -1
        
        for user_data in users_data:
            user_id, user_results, save_path, error_msg = process_user_wrapper(
                (user_data, args, gpu_id, output_dir, timestamp)
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
