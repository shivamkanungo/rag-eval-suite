"""
RAGAS evaluation wrapper — computes the four core RAG metrics:
  • answer_relevancy    — how relevant is the answer to the question?
  • faithfulness        — is the answer grounded in the context? (hallucination)
  • context_precision   — are the retrieved chunks precise (not noisy)?
  • context_recall      — were all relevant chunks retrieved?

LangSmith feedback is posted for each evaluation run when tracing is enabled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    answer_relevancy: float
    faithfulness: float
    context_precision: float
    context_recall: float
    question: str
    answer: str
    raw_scores: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate(self) -> float:
        """Harmonic mean of all four metrics."""
        scores = [
            self.answer_relevancy,
            self.faithfulness,
            self.context_precision,
            self.context_recall,
        ]
        valid = [s for s in scores if s is not None and s >= 0]
        if not valid:
            return 0.0
        return len(valid) / sum(1 / s for s in valid if s > 0)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> RAGASResult:
    """
    Run RAGAS evaluation for a single question-answer pair.

    Args:
        question:     The user query.
        answer:       The generated answer.
        contexts:     List of retrieved context strings.
        ground_truth: Optional reference answer (needed for context_recall).

    Returns:
        RAGASResult with all metric scores.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.error("RAGAS not installed: %s", e)
        return _fallback_result(question, answer)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics = [answer_relevancy, faithfulness, context_precision]

    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics.append(context_recall)

    try:
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()
        row = df.iloc[0]

        return RAGASResult(
            answer_relevancy=float(row.get("answer_relevancy", -1)),
            faithfulness=float(row.get("faithfulness", -1)),
            context_precision=float(row.get("context_precision", -1)),
            context_recall=float(row.get("context_recall", -1) if ground_truth else -1),
            question=question,
            answer=answer,
            raw_scores=row.to_dict(),
        )
    except Exception as e:
        logger.exception("RAGAS evaluation failed: %s", e)
        return _fallback_result(question, answer)


def evaluate_batch(
    samples: list[dict],
) -> list[RAGASResult]:
    """
    Evaluate a batch of samples.

    Each sample dict must have:
      - question (str)
      - answer (str)
      - contexts (list[str])
      - ground_truth (str, optional)
    """
    results = []
    for sample in samples:
        result = evaluate_single(
            question=sample["question"],
            answer=sample["answer"],
            contexts=sample["contexts"],
            ground_truth=sample.get("ground_truth"),
        )
        results.append(result)
    return results


def _fallback_result(question: str, answer: str) -> RAGASResult:
    """Return a zeroed result when RAGAS is unavailable."""
    return RAGASResult(
        answer_relevancy=-1,
        faithfulness=-1,
        context_precision=-1,
        context_recall=-1,
        question=question,
        answer=answer,
        raw_scores={"error": "RAGAS evaluation unavailable"},
    )
