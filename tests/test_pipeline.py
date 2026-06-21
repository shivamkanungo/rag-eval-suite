"""
Integration tests for the RAG pipeline.
Run with: pytest tests/ -v

These tests use real API calls — set OPENAI_API_KEY in environment.
Mark tests as integration: pytest tests/ -v -m integration
"""
import os
import pytest


# ─── Unit: Document Loader ────────────────────────────────────────────────────

def test_document_chunking():
    from backend.utils.document_loader import load_from_text

    long_text = "This is a sentence. " * 100
    chunks = load_from_text(long_text, source="test")
    assert len(chunks) > 1, "Long text should produce multiple chunks"
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test"


def test_chunk_metadata():
    from backend.utils.document_loader import load_from_text

    chunks = load_from_text("Hello world.", source="test_source")
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "test_source"
    assert chunks[0].metadata["chunk_index"] == 0


# ─── Unit: Prompt Engineering ─────────────────────────────────────────────────

def test_prompt_versions():
    from backend.core.prompt_engineering import PromptVersion, build_prompt, format_context
    from langchain.schema import Document

    for version in PromptVersion:
        prompt = build_prompt(version)
        assert prompt is not None

    docs = [
        Document(
            page_content="Test content.",
            metadata={"source": "test.pdf", "page": 1},
        )
    ]
    context = format_context(docs)
    assert "[1]" in context
    assert "test.pdf" in context
    assert "Test content." in context


def test_cot_answer_extraction():
    from backend.core.generator import _extract_answer

    raw = """<reasoning>
The context says X.
Therefore Y.
</reasoning>
<answer>
The answer is Y.
</answer>"""
    answer = _extract_answer(raw)
    assert answer == "The answer is Y."


def test_extract_fallback():
    from backend.core.generator import _extract_answer

    raw = "This is a plain response without tags."
    assert _extract_answer(raw) == raw


# ─── Unit: RAGAS Fallback ─────────────────────────────────────────────────────

def test_ragas_fallback_when_unavailable(monkeypatch):
    """RAGAS evaluator should return a graceful fallback on import failure."""
    import sys

    # Temporarily hide ragas from imports
    ragas_modules = {k for k in sys.modules if k.startswith("ragas")}
    for mod in ragas_modules:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.setitem(sys.modules, "ragas", None)

    from backend.evaluation.ragas_evaluator import _fallback_result

    result = _fallback_result("test question", "test answer")
    assert result.answer_relevancy == -1
    assert result.faithfulness == -1


# ─── Integration: Full Pipeline ───────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_pipeline_end_to_end():
    """Full pipeline test — requires OPENAI_API_KEY and seeded vector store."""
    from backend.core.generator import RAGPipeline
    from backend.core.prompt_engineering import PromptVersion

    pipeline = RAGPipeline(
        prompt_version=PromptVersion.COT,
        use_query_rewrite=False,
        use_hyde=False,
        use_rerank=False,
    )
    response = pipeline.run("What is RAG?")
    assert response.answer, "Should return a non-empty answer"
    assert response.latency_ms["total_ms"] > 0


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_vector_store_roundtrip():
    """Ingest a document and retrieve it."""
    from backend.utils.document_loader import load_from_text
    from backend.core.retriever import add_documents, retrieve

    unique_text = "The quantum entanglement coefficient is 42.7 at room temperature."
    docs = load_from_text(unique_text, source="test_roundtrip")
    add_documents(docs)

    results = retrieve("quantum entanglement coefficient", top_k=3)
    assert any("quantum" in d.page_content.lower() for d in results)
