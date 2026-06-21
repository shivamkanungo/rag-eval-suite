"""
Vector store management — ChromaDB-backed retriever with MMR and
optional HyDE (Hypothetical Document Embeddings) for query expansion.
"""
import logging
from functools import lru_cache

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_vector_store() -> Chroma:
    return Chroma(
        client=_get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def add_documents(documents: list[Document]) -> list[str]:
    """Upsert documents into ChromaDB. Returns inserted IDs."""
    store = get_vector_store()
    ids = [doc.metadata["chunk_id"] for doc in documents]
    store.add_documents(documents, ids=ids)
    logger.info("Upserted %d documents into vector store", len(documents))
    return ids


def retrieve(
    query: str,
    top_k: int | None = None,
    use_hyde: bool = False,
) -> list[Document]:
    """
    Retrieve relevant documents.

    Args:
        query:    The user query.
        top_k:    Number of results (defaults to settings).
        use_hyde: If True, generate a hypothetical answer to expand retrieval.
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval

    if use_hyde:
        query = _hyde_expand(query)
        logger.info("HyDE-expanded query: %s", query[:100])

    store = get_vector_store()
    retriever = store.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    docs = retriever.invoke(query)
    logger.info("Retrieved %d docs for query: '%s'", len(docs), query[:60])
    return docs


def get_collection_stats() -> dict:
    """Return basic stats about the vector store."""
    client = _get_chroma_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        return {"document_count": col.count(), "collection": COLLECTION_NAME}
    except Exception:
        return {"document_count": 0, "collection": COLLECTION_NAME}


def _hyde_expand(query: str) -> str:
    """
    HyDE: ask the LLM to write a hypothetical answer,
    then use that answer as the search query.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=settings.openai_api_key,
    )
    prompt = (
        "Write a short, factual paragraph that would be the ideal answer to the "
        f"following question. Answer only, no preamble.\n\nQuestion: {query}"
    )
    return llm.invoke(prompt).content
