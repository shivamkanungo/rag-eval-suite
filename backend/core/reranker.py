"""
Cross-encoder re-ranker — takes retrieved docs and re-scores them
using an LLM-based relevance judge, then returns top-k.

This is the "re-ranking" stage that improves precision after
the initial vector search recall pass.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from backend.config import get_settings

logger = logging.getLogger(__name__)


class RelevanceScore(BaseModel):
    score: float  # 0.0 – 1.0
    reasoning: str


_JUDGE_SYSTEM = """You are a relevance judge. Given a question and a context
passage, rate how relevant the passage is to answering the question.

Respond ONLY with valid JSON:
{"score": <float 0-1>, "reasoning": "<one sentence>"}

Score guide:
  1.0 — directly answers the question
  0.7 — highly relevant, contains key info
  0.4 — somewhat related, tangential
  0.1 — mostly irrelevant
  0.0 — completely off-topic
"""

_JUDGE_HUMAN = """Question: {question}

Passage:
{passage}

JSON:"""


def rerank(
    query: str,
    docs: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """
    Re-rank documents by LLM-judged relevance. Returns top_k docs
    sorted by descending relevance score.
    """
    if not docs:
        return []

    settings = get_settings()
    k = top_k or settings.rerank_top_k
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    scored: list[tuple[float, Document]] = []

    for doc in docs:
        prompt_text = _JUDGE_HUMAN.format(
            question=query, passage=doc.page_content[:800]
        )
        try:
            raw = llm.invoke(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt_text},
                ]
            ).content
            import json
            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            doc.metadata["rerank_score"] = score
            doc.metadata["rerank_reasoning"] = data.get("reasoning", "")
            scored.append((score, doc))
        except Exception as e:
            logger.warning("Rerank failed for chunk %s: %s", doc.metadata.get("chunk_id"), e)
            doc.metadata["rerank_score"] = 0.5
            scored.append((0.5, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in scored[:k]]
    logger.info("Reranked %d -> %d docs for query: '%s'", len(docs), len(result), query[:60])
    return result
