"""
FastAPI application — REST API for the RAG evaluation suite.
"""
import logging
import time

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import get_settings
from backend.core.generator import RAGPipeline
from backend.core.prompt_engineering import PromptVersion
from backend.core.retriever import add_documents, get_collection_stats
from backend.evaluation.ragas_evaluator import evaluate_single
from backend.utils.document_loader import load_from_text, load_from_upload

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="RAG Evaluation Suite",
    description="Vector search, prompt engineering, and RAGAS/LangSmith evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared pipeline instance ─────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            prompt_version=PromptVersion.FEW_SHOT_COT,
            use_query_rewrite=True,
            use_hyde=False,
            use_rerank=True,
        )
    return _pipeline


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    ground_truth: str | None = None
    run_evaluation: bool = True
    prompt_version: str = "few_shot_cot_v3"
    use_hyde: bool = False


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class EvalOnlyRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    stats = get_collection_stats()
    return {
        "status": "ok",
        "model": settings.llm_model,
        "langsmith_enabled": settings.langchain_tracing_v2,
        **stats,
    }


@app.post("/query")
def query(req: QueryRequest):
    """
    Main RAG query endpoint.
    Runs the full pipeline and optionally evaluates with RAGAS.
    """
    try:
        version_map = {v.value: v for v in PromptVersion}
        version = version_map.get(req.prompt_version, PromptVersion.FEW_SHOT_COT)

        pipeline = RAGPipeline(
            prompt_version=version,
            use_query_rewrite=True,
            use_hyde=req.use_hyde,
            use_rerank=True,
        )
        response = pipeline.run(req.question)
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "answer": response.answer,
        "rewritten_query": response.rewritten_query,
        "prompt_version": response.prompt_version,
        "latency_ms": response.latency_ms,
        "metadata": response.metadata,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "rerank_score": doc.metadata.get("rerank_score", None),
                "content_preview": doc.page_content[:200],
            }
            for doc in response.reranked_docs
        ],
        "evaluation": None,
    }

    if req.run_evaluation and response.reranked_docs:
        try:
            eval_result = evaluate_single(
                question=req.question,
                answer=response.answer,
                contexts=[d.page_content for d in response.reranked_docs],
                ground_truth=req.ground_truth,
            )
            result["evaluation"] = {
                "answer_relevancy": eval_result.answer_relevancy,
                "faithfulness": eval_result.faithfulness,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "aggregate": eval_result.aggregate,
            }
        except Exception as e:
            logger.warning("Evaluation failed (non-fatal): %s", e)

    return result


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    """Ingest plain text into the vector store."""
    try:
        docs = load_from_text(req.text, source=req.source)
        ids = add_documents(docs)
        return {"ingested": len(ids), "source": req.source, "chunk_ids": ids}
    except Exception as e:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile):
    """Ingest a PDF or text file."""
    try:
        content = await file.read()
        docs = load_from_upload(content, file.filename or "upload")
        ids = add_documents(docs)
        return {
            "filename": file.filename,
            "ingested": len(ids),
            "chunk_ids": ids[:5],  # preview
        }
    except Exception as e:
        logger.exception("File ingest error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate_only(req: EvalOnlyRequest):
    """Run RAGAS evaluation on a pre-existing answer without calling the LLM."""
    try:
        result = evaluate_single(
            question=req.question,
            answer=req.answer,
            contexts=req.contexts,
            ground_truth=req.ground_truth,
        )
        return {
            "answer_relevancy": result.answer_relevancy,
            "faithfulness": result.faithfulness,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "aggregate": result.aggregate,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Collection statistics for the dashboard."""
    return get_collection_stats()
