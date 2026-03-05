from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Literal, Any, Union
import numpy as np
import os
from pydantic import BaseModel, Field
import pickle
from openai import OpenAI


class FlexibleEmbeddingRetriever:
    """
    Extended retriever supporting multiple embedding models.
    
    Supported models:
    - SentenceTransformer models: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', etc.
    - Contriever: 'facebook/contriever'
    - Stella: 'dunzhang/stella_en_1.5B_v5'
    - GTE: 'Alibaba-NLP/gte-Qwen2-7B-instruct'
    - OpenAI: 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    - BM25: 'bm25' (sparse retrieval)
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 model_type: Literal['sentence-transformer', 'contriever', 'stella', 'gte', 'openai', 'bm25'] = 'sentence-transformer',
                 device: str = None,
                 cache_dir: str = None,
                 api_key: str = None,
                 base_url: str = None):
        """
        Initialize the flexible retriever.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of embedding model
            device: Device to run the model on (cuda/cpu)
            cache_dir: Cache directory for model files
            api_key: API key for OpenAI (required if model_type is 'openai')
            base_url: Base URL for OpenAI API (optional)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.base_url = base_url
        
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}
        self.bm25 = None  # BM25 index
        
        # Initialize model based on type
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model based on type"""
        if self.model_type == 'sentence-transformer':
            # Standard SentenceTransformer models
            self.model = SentenceTransformer(self.model_name)
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
        
        elif self.model_type == 'contriever':
            # Facebook Contriever
            self.tokenizer = AutoTokenizer.from_pretrained(
                'facebook/contriever',
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                'facebook/contriever',
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
        
        elif self.model_type == 'stella':
            # Stella model
            model_dir = self.cache_dir + "/dunzhang_stella_en_1.5B_v5"
            self.vector_dim = 1024
            vector_linear_directory = f"2_Dense_{self.vector_dim}"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
            self.model.eval()
            
            # Load vector linear layer
            self.vector_linear = torch.nn.Linear(
                in_features=self.model.config.hidden_size, 
                out_features=self.vector_dim
            ).to(self.device)
            
            if os.path.exists(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")):
                vector_linear_dict = {
                    k.replace("linear.", ""): v for k, v in
                    torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
                }
                self.vector_linear.load_state_dict(vector_linear_dict)
            self.vector_linear.to(self.device)
        
        elif self.model_type == 'gte':
            # GTE model
            self.tokenizer = AutoTokenizer.from_pretrained(
                'Alibaba-NLP/gte-Qwen2-7B-instruct',
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                'Alibaba-NLP/gte-Qwen2-7B-instruct',
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.model.eval()
        
        elif self.model_type == 'bm25':
            # BM25 doesn't need model initialization
            # Index will be built when documents are added
            self.model = None
        
        elif self.model_type == 'openai':
            # OpenAI embeddings API
            if not self.api_key:
                # Try to get from environment variable
                self.api_key = os.getenv('OPENAI_API_KEY')
                if not self.api_key:
                    raise ValueError("API key required for OpenAI embeddings. Provide api_key parameter or set OPENAI_API_KEY environment variable.")
            
            # Initialize OpenAI client
            client_kwargs = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            self.model = OpenAI(**client_kwargs)
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _encode_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using SentenceTransformer"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def _encode_contriever(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts using Contriever"""
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                all_embeddings.append(embeddings.numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _encode_stella(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts using Stella"""
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                input_data = self.tokenizer(
                    batch, 
                    padding="longest", 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                )
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                attention_mask = input_data["attention_mask"]
                last_hidden_state = self.model(**input_data)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                embeddings = normalize(self.vector_linear(embeddings).detach().cpu())
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _encode_gte(self, texts: List[str], batch_size: int = 1, task: str = None) -> np.ndarray:
        """Encode texts using GTE"""
        def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruction: {task_description}\nQuery: {query}'
        
        # Default task if not provided
        if task is None:
            task = 'Given a query about personal information, retrieve relevant chat history that answer the query.'
        
        # Add instruction to first text (query) only
        formatted_texts = [get_detailed_instruct(task, texts[0])] + texts[1:] if len(texts) > 0 else texts
        
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(formatted_texts), batch_size):
                batch = formatted_texts[i:i + batch_size]
                batch_dict = self.tokenizer(
                    batch, 
                    max_length=8192, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _encode_openai(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Encode texts using OpenAI's embedding API.
        
        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode in one API call (max 2048 for OpenAI)
            
        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.model.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {str(e)}")
                raise
        
        return np.array(all_embeddings)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using the selected model.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        if self.model_type == 'sentence-transformer':
            return self._encode_sentence_transformer(texts)
        elif self.model_type == 'contriever':
            return self._encode_contriever(texts)
        elif self.model_type == 'stella':
            return self._encode_stella(texts)
        elif self.model_type == 'gte':
            return self._encode_gte(texts)
        elif self.model_type == 'openai':
            return self._encode_openai(texts)
        elif self.model_type == 'bm25':
            # BM25 doesn't use embeddings
            return None
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        if not self.corpus:
            self.corpus = documents
            if self.model_type == 'bm25':
                # Tokenize documents for BM25
                tokenized_corpus = [doc.split(" ") for doc in documents]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.embeddings = self.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            if self.model_type == 'bm25':
                # Rebuild BM25 index with all documents
                tokenized_corpus = [doc.split(" ") for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                new_embeddings = self.encode(documents)
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Search for similar documents using cosine similarity or BM25.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of top-k document indices
        """
        if not self.corpus:
            return []
        
        k = min(k, len(self.corpus))
        
        if self.model_type == 'bm25':
            # Use BM25 scoring
            tokenized_query = query.split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(scores)[-k:][::-1]
        else:
            # Use embedding-based cosine similarity
            query_embedding = self.encode([query])[0]
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, model_type: str, 
                              device: str = None, cache_dir: str = None,
                              api_key: str = None, base_url: str = None) -> 'FlexibleEmbeddingRetriever':
        """
        Load retriever state from memory objects.
        
        Args:
            memories: Dictionary of memory objects
            model_name: Name of the embedding model
            model_type: Type of embedding model
            device: Device to run on
            cache_dir: Cache directory for models
            api_key: API key for OpenAI (if using openai model_type)
            base_url: Base URL for OpenAI API (optional)
            
        Returns:
            Initialized retriever with memories
        """
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
        
        # Create and initialize retriever
        retriever = cls(model_name=model_name, model_type=model_type, device=device, 
                       cache_dir=cache_dir, api_key=api_key, base_url=base_url)
        retriever.add_documents(all_docs)
        return retriever

    def save(self, retriever_cache_file:str, retriever_cache_embeddings_file: str):
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self