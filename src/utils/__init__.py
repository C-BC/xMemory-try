"""
Utility Modules
"""

from .llm_client import LLMClient as OpenAILLMClient
from .llm_client_openllm import LLMClient as HFLLMClient
from .embedding_client import EmbeddingClient
from .performance import PerformanceOptimizer

# Known OpenAI model prefixes — these require the OpenAI API client
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "text-", "chatgpt-")


def LLMClient(api_key: str = "", model: str = "gpt-4o-mini", **kwargs):
    """Factory that returns the right LLM client based on model name.

    - Models starting with known OpenAI prefixes use the OpenAI API client.
    - All other models (e.g. Qwen/Qwen3-8B) use the local HuggingFace client.
    """
    if any(model.lower().startswith(p) for p in _OPENAI_PREFIXES):
        return OpenAILLMClient(api_key=api_key, model=model, **kwargs)
    return HFLLMClient(api_key=api_key, model=model, **kwargs)


__all__ = [
    "LLMClient",
    "OpenAILLMClient",
    "HFLLMClient",
    "EmbeddingClient",
    "PerformanceOptimizer",
] 