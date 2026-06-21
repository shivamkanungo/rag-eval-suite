"""
Seed the vector store with RAG/ML knowledge base content.
Run once before starting the server for demo purposes.

Usage:
    cd backend
    python -m scripts.seed_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED_DOCUMENTS = [
    {
        "source": "rag_fundamentals",
        "text": """
Retrieval-Augmented Generation (RAG) is a framework that enhances large language models
by grounding their outputs in retrieved external knowledge. Rather than relying solely on
parameters baked during training, RAG systems dynamically pull relevant documents from a
knowledge base at inference time, providing the LLM with factual context that informs its
generated response.

The RAG pipeline consists of three main stages:
1. Indexing — Documents are chunked, embedded into vectors, and stored in a vector database.
2. Retrieval — A user query is embedded and used to find the most semantically similar
   document chunks via cosine similarity or maximum marginal relevance (MMR).
3. Generation — Retrieved chunks are injected into the prompt as context, and the LLM
   generates a grounded, factual response.

RAG significantly reduces hallucinations because the model has explicit evidence to
reference. It also makes knowledge updatable without retraining — simply re-index new
documents and the system immediately has access to the latest information.
""",
    },
    {
        "source": "vector_search_deep_dive",
        "text": """
Vector search (also called semantic search) converts text into high-dimensional
embedding vectors and retrieves documents by computing similarity in that embedding
space. Unlike keyword search (BM25, TF-IDF), vector search captures semantic meaning —
the query "car engine malfunction" will match "automobile motor failure" even without
shared keywords.

Key concepts:
- Embeddings: Dense numerical representations of text produced by models like
  OpenAI text-embedding-3-small (1536 dimensions) or sentence-transformers.
- Cosine similarity: Measures the angle between two vectors; values range from -1 to 1
  where 1 means identical direction.
- Approximate Nearest Neighbor (ANN): Algorithms like HNSW (used by ChromaDB) that
  trade slight accuracy for massive speed improvements at scale.
- Maximum Marginal Relevance (MMR): A retrieval strategy that balances relevance and
  diversity, avoiding returning near-duplicate chunks.

ChromaDB is a popular open-source vector store that persists embeddings locally using
SQLite and HNSW, making it ideal for development and mid-scale production workloads.
""",
    },
    {
        "source": "ragas_evaluation_guide",
        "text": """
RAGAS (Retrieval-Augmented Generation Assessment) is an evaluation framework that
provides reference-free metrics for assessing RAG pipeline quality.

Core metrics:

1. Answer Relevancy — Measures how relevant the generated answer is to the input
   question. Computed by generating synthetic questions from the answer and checking
   if they are semantically similar to the original question. Range: 0–1.

2. Faithfulness — Measures whether the answer is factually grounded in the retrieved
   context. The LLM extracts claims from the answer and checks each claim against the
   context. Hallucinated claims reduce this score. Range: 0–1.

3. Context Precision — Measures the signal-to-noise ratio of retrieved chunks.
   Checks whether the top-ranked chunks are actually relevant. Higher precision means
   fewer irrelevant chunks contaminate the context window. Range: 0–1.

4. Context Recall — Measures how much of the ground-truth answer can be attributed to
   the retrieved context. Requires a reference answer. Range: 0–1.

A balanced RAG system should score above 0.8 on all four metrics. In practice,
faithfulness and answer relevancy are the most actionable since they do not require
ground-truth labels.
""",
    },
    {
        "source": "langsmith_observability",
        "text": """
LangSmith is Anthropic/LangChain's observability and evaluation platform for LLM
applications. It provides full-chain tracing, allowing developers to inspect every step
of a LangChain pipeline — from prompt construction to LLM response to tool calls.

Key capabilities:
- Tracing: Every chain invocation creates a run tree with input/output, latency, and
  token cost at each node.
- Feedback: Human or automated scores (like RAGAS metrics) can be posted to runs,
  enabling A/B testing of prompt versions.
- Datasets: Build curated eval datasets from production traces.
- Experiments: Run batch evaluations comparing pipeline variants side-by-side.
- Monitoring: Set up dashboards tracking p50/p95 latency, error rates, and quality
  metrics over time.

LangSmith integrates with LangChain via environment variables:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your key>
  LANGCHAIN_PROJECT=<project name>
No code changes are required — tracing is automatic when these are set.
""",
    },
    {
        "source": "prompt_engineering_guide",
        "text": """
Prompt engineering is the practice of designing input templates that reliably elicit
high-quality, consistent outputs from language models.

Techniques with demonstrated impact on RAG quality:

1. Chain-of-Thought (CoT) prompting — Instructing the model to "think step by step"
   before giving a final answer. CoT significantly reduces hallucinations by forcing the
   model to surface its reasoning, making errors visible and correctable.

2. Few-shot examples — Including 2–5 example question/answer pairs in the system prompt
   calibrates the model's output format, tone, and level of detail.

3. Explicit grounding instructions — Telling the model explicitly "answer only from the
   provided context; say 'I don't know' if insufficient" reduces confabulation by 20–40%.

4. Query rewriting — Using an LLM to rephrase the user's raw question into a more
   retrieval-optimized form before embedding. This handles typos, informal language, and
   ambiguous references.

5. HyDE (Hypothetical Document Embeddings) — Generating a hypothetical ideal answer and
   using its embedding for retrieval. Particularly effective when queries are short or
   when the document corpus uses technical vocabulary.

Prompt version control is important: each template should be tagged with a version string
so that LangSmith experiments can compare metrics across versions.
""",
    },
    {
        "source": "reranking_techniques",
        "text": """
Re-ranking is a post-retrieval stage that re-scores retrieved documents using a more
powerful relevance model before passing them to the generator.

Why re-rank? Vector search optimizes for semantic similarity, not task-specific
relevance. A chunk about "neural network training" might score high for the query
"how to reduce overfitting" because they share embedding space, even if the chunk
doesn't directly address the question. Re-ranking adds a precision pass.

Re-ranking approaches:

1. Cross-encoder re-ranking — A model (or LLM-as-judge) scores each (query, chunk)
   pair jointly. Slower but significantly more accurate than bi-encoder similarity.

2. LLM-as-judge — Prompt an LLM to rate relevance on a 0–1 scale with brief reasoning.
   Flexible and easy to implement, though adds latency and cost.

3. Cohere Rerank API — A dedicated reranking endpoint that scores batches of documents
   against a query in a single API call. Fast, accurate, cost-effective at scale.

4. ColBERT — A late-interaction model that computes fine-grained token-level interactions.
   State-of-the-art accuracy but requires local model hosting.

In typical RAG pipelines, retrieve 10–20 candidates with vector search, then re-rank and
keep the top 3–5. This "retrieve-then-rerank" pattern consistently outperforms using only
one stage.
""",
    },
]


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("seed")

    from backend.utils.document_loader import load_from_text
    from backend.core.retriever import add_documents, get_collection_stats

    logger.info("Seeding vector store with %d documents...", len(SEED_DOCUMENTS))

    all_ids = []
    for doc in SEED_DOCUMENTS:
        chunks = load_from_text(doc["text"], source=doc["source"])
        ids = add_documents(chunks)
        all_ids.extend(ids)
        logger.info("  ✓ %s → %d chunks", doc["source"], len(ids))

    stats = get_collection_stats()
    logger.info("Seed complete. Total docs in store: %d", stats["document_count"])


if __name__ == "__main__":
    main()
