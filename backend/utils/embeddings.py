"""
Embeddings wrapper — singleton OpenAI embedder with retry logic.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        openai_api_key=settings.openai_api_key,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_query(text: str) -> list[float]:
    """Embed a single query string with retry."""
    return get_embeddings().embed_query(text)
