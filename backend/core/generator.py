"""
RAG Pipeline — orchestrates retrieve → rerank → generate.
Integrates LangSmith tracing transparently.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from langchain.schema import Document
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.core.prompt_engineering import (
    PromptVersion,
    build_prompt,
    build_rewrite_prompt,
    format_context,
)
from backend.core.retriever import retrieve
from backend.core.reranker import rerank

logger = logging.getLogger(__name__)


# ─── Output Schema ────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    raw_answer: str           # Full LLM output including CoT
    query: str
    rewritten_query: str
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    context_used: str
    prompt_version: str
    latency_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline with:
      1. Optional query rewriting
      2. Vector retrieval (with optional HyDE)
      3. LLM re-ranking
      4. Chain-of-thought generation
      5. LangSmith tracing (auto-enabled when LANGCHAIN_TRACING_V2=true)
    """

    def __init__(
        self,
        prompt_version: PromptVersion = PromptVersion.FEW_SHOT_COT,
        use_query_rewrite: bool = True,
        use_hyde: bool = False,
        use_rerank: bool = True,
    ):
        settings = get_settings()
        self.settings = settings
        self.prompt_version = prompt_version
        self.use_query_rewrite = use_query_rewrite
        self.use_hyde = use_hyde
        self.use_rerank = use_rerank

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )

        # Configure LangSmith if enabled
        if settings.langchain_tracing_v2 and settings.langchain_api_key != "disabled":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            logger.info("LangSmith tracing enabled for project: %s", settings.langchain_project)

    def run(self, query: str) -> RAGResponse:
        latency: dict[str, float] = {}
        total_start = time.perf_counter()

        # 1. Query rewriting
        t0 = time.perf_counter()
        rewritten = self._rewrite_query(query) if self.use_query_rewrite else query
        latency["query_rewrite_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Retrieval
        t0 = time.perf_counter()
        retrieved = retrieve(rewritten, use_hyde=self.use_hyde)
        latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Re-ranking
        t0 = time.perf_counter()
        reranked = rerank(rewritten, retrieved) if self.use_rerank else retrieved[: self.settings.rerank_top_k]
        latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Context formatting
        context = format_context(reranked)

        # 5. Generation
        t0 = time.perf_counter()
        raw_answer = self._generate(rewritten, context)
        latency["generation_ms"] = (time.perf_counter() - t0) * 1000

        latency["total_ms"] = (time.perf_counter() - total_start) * 1000

        # Parse structured answer from CoT output
        answer = _extract_answer(raw_answer)

        logger.info(
            "RAG complete | query='%s' | docs=%d | latency=%.0fms",
            query[:60],
            len(reranked),
            latency["total_ms"],
        )

        return RAGResponse(
            answer=answer,
            raw_answer=raw_answer,
            query=query,
            rewritten_query=rewritten,
            retrieved_docs=retrieved,
            reranked_docs=reranked,
            context_used=context,
            prompt_version=self.prompt_version.value,
            latency_ms=latency,
            metadata={
                "model": self.settings.llm_model,
                "use_hyde": self.use_hyde,
                "use_rerank": self.use_rerank,
                "num_retrieved": len(retrieved),
                "num_reranked": len(reranked),
            },
        )

    def _rewrite_query(self, query: str) -> str:
        prompt = build_rewrite_prompt()
        chain = prompt | self.llm
        rewritten = chain.invoke({"question": query}).content.strip()
        logger.debug("Query rewrite: '%s' → '%s'", query, rewritten)
        return rewritten

    def _generate(self, query: str, context: str) -> str:
        prompt = build_prompt(self.prompt_version)
        chain = prompt | self.llm
        return chain.invoke({"question": query, "context": context}).content


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_answer(raw: str) -> str:
    """
    Extract text inside <answer>...</answer> tags if present,
    otherwise return the full response.
    """
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip any <reasoning> block
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.DOTALL)
    return cleaned.strip()
