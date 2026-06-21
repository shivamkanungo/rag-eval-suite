"""
Batch evaluation runner — evaluates a JSON dataset through the RAG pipeline
and outputs a RAGAS report.

Usage:
    python -m scripts.run_evaluation \
        --dataset data/eval_dataset.json \
        --output results/ragas_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS batch evaluation")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    parser.add_argument("--output", default="results/ragas_report.json", help="Output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    samples = json.loads(dataset_path.read_text())
    if args.limit:
        samples = samples[: args.limit]

    logger.info("Evaluating %d samples...", len(samples))

    # Lazy import — avoids loading heavy deps until needed
    from backend.core.generator import RAGPipeline
    from backend.core.prompt_engineering import PromptVersion
    from backend.evaluation.ragas_evaluator import evaluate_single

    pipeline = RAGPipeline(prompt_version=PromptVersion.FEW_SHOT_COT)

    results = []
    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        ground_truth = sample.get("ground_truth")
        logger.info("[%d/%d] %s", i, len(samples), question[:60])

        t0 = time.perf_counter()
        response = pipeline.run(question)
        elapsed = (time.perf_counter() - t0) * 1000

        eval_result = evaluate_single(
            question=question,
            answer=response.answer,
            contexts=[d.page_content for d in response.reranked_docs],
            ground_truth=ground_truth,
        )

        results.append(
            {
                "question": question,
                "answer": response.answer,
                "rewritten_query": response.rewritten_query,
                "latency_ms": elapsed,
                "answer_relevancy": eval_result.answer_relevancy,
                "faithfulness": eval_result.faithfulness,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "aggregate": eval_result.aggregate,
            }
        )

    # Summary stats
    def avg(key):
        vals = [r[key] for r in results if r[key] >= 0]
        return round(statistics.mean(vals), 4) if vals else -1

    summary = {
        "total_samples": len(results),
        "averages": {
            "answer_relevancy": avg("answer_relevancy"),
            "faithfulness": avg("faithfulness"),
            "context_precision": avg("context_precision"),
            "context_recall": avg("context_recall"),
            "aggregate": avg("aggregate"),
            "latency_ms": avg("latency_ms"),
        },
        "samples": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    logger.info("Report saved to %s", output_path)
    logger.info("Average aggregate score: %.3f", summary["averages"]["aggregate"])


if __name__ == "__main__":
    main()
