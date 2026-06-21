from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI
from backend.config import get_settings

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
        scores = [self.answer_relevancy, self.faithfulness, self.context_precision]
        valid = [s for s in scores if s >= 0]
        return round(sum(valid) / len(valid), 4) if valid else 0.0


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> RAGASResult:
    try:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        context_text = "\n\n".join(contexts[:3])

        prompt = f"""You are an evaluation judge for a RAG system. Score the following on three metrics, each from 0.0 to 1.0.

Question: {question}
Answer: {answer}
Context: {context_text}

Scoring rules:
- answer_relevancy: Does the answer directly address the question? (1.0 = perfectly relevant)
- faithfulness: Is the answer fully supported by the context with no hallucination? (1.0 = fully grounded)
- context_precision: Does the context contain useful information for answering the question? (1.0 = highly precise)

Respond ONLY with valid JSON like this:
{{"answer_relevancy": 0.9, "faithfulness": 0.85, "context_precision": 0.8}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)

        return RAGASResult(
            answer_relevancy=float(scores.get("answer_relevancy", -1)),
            faithfulness=float(scores.get("faithfulness", -1)),
            context_precision=float(scores.get("context_precision", -1)),
            context_recall=-1,
            question=question,
            answer=answer,
            raw_scores=scores,
        )
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return _fallback_result(question, answer)


def evaluate_batch(samples: list[dict]) -> list[RAGASResult]:
    return [evaluate_single(**s) for s in samples]


def _fallback_result(question: str, answer: str) -> RAGASResult:
    return RAGASResult(
        answer_relevancy=-1,
        faithfulness=-1,
        context_precision=-1,
        context_recall=-1,
        question=question,
        answer=answer,
        raw_scores={"error": "Evaluation failed"},
    )