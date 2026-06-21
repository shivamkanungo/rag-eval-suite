"""
LangSmith integration — post evaluation feedback and retrieve run data.
Gracefully no-ops when LangSmith is disabled.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from backend.config import get_settings
from backend.evaluation.ragas_evaluator import RAGASResult

logger = logging.getLogger(__name__)


def _get_client():
    """Return LangSmith client or None if disabled."""
    settings = get_settings()
    if not settings.langchain_tracing_v2 or settings.langchain_api_key == "disabled":
        return None
    try:
        from langsmith import Client
        return Client(api_key=settings.langchain_api_key)
    except Exception as e:
        logger.warning("LangSmith client init failed: %s", e)
        return None


def post_eval_feedback(run_id: str, result: RAGASResult) -> bool:
    """
    Post RAGAS scores as feedback on a LangSmith run.

    Returns True if feedback was posted successfully.
    """
    client = _get_client()
    if not client:
        logger.debug("LangSmith disabled — skipping feedback post")
        return False

    metrics = {
        "answer_relevancy": result.answer_relevancy,
        "faithfulness": result.faithfulness,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
        "aggregate": result.aggregate,
    }

    success = True
    for key, score in metrics.items():
        if score < 0:
            continue
        try:
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=f"RAGAS auto-eval: {key}={score:.3f}",
            )
        except Exception as e:
            logger.warning("Failed to post feedback '%s': %s", key, e)
            success = False

    return success


def get_recent_runs(limit: int = 20) -> list[dict]:
    """Fetch recent runs from LangSmith for the dashboard."""
    client = _get_client()
    if not client:
        return []

    try:
        settings = get_settings()
        runs = list(
            client.list_runs(
                project_name=settings.langchain_project,
                run_type="chain",
                limit=limit,
            )
        )
        return [
            {
                "id": str(r.id),
                "name": r.name,
                "start_time": r.start_time.isoformat() if r.start_time else None,
                "latency_ms": r.total_tokens,  # proxy
                "status": r.status,
            }
            for r in runs
        ]
    except Exception as e:
        logger.warning("Failed to fetch LangSmith runs: %s", e)
        return []
