"""
Central settings — all env vars validated here via Pydantic.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # RAG
    top_k_retrieval: int = 5
    rerank_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 64

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = "disabled"
    langchain_project: str = "rag-eval-suite"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # RAGAS
    ragas_run_config_timeout: int = 120
    eval_batch_size: int = 10

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
