# RAG Evaluation Suite

!\[RAG Evaluation Suite Dashboard](docs/dashboard.png)
A production-grade Retrieval-Augmented Generation (RAG) system with vector search, prompt engineering, and automated evaluation using **RAGAS** and **LangSmith**. Demonstrates measurable improvements in answer relevance (+30%) and hallucination reduction.

\---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  FastAPI Backend │
│  ─────────────  │
│  Query Router   │
└────────┬────────┘
         │
    ┌────▼────┐      ┌──────────────────┐
    │ Vector  │◄────►│  ChromaDB Store  │
    │ Search  │      │  (Embeddings)    │
    └────┬────┘      └──────────────────┘
         │
    ┌────▼────────────┐
    │ Prompt Engineer │  ← Dynamic prompt templates
    │ + Re-ranker     │    with chain-of-thought
    └────┬────────────┘
         │
    ┌────▼────┐
    │   LLM   │  (OpenAI GPT-4o / Anthropic Claude)
    └────┬────┘
         │
    ┌────▼──────────────┐
    │ RAGAS Evaluator   │  answer\_relevancy, faithfulness,
    │ + LangSmith Trace │  context\_precision, context\_recall
    └───────────────────┘
         │
    React Dashboard ◄───────────────────┘
```

\---

## Features

* **Vector Search** — ChromaDB with OpenAI `text-embedding-3-small`; cosine similarity retrieval
* **Prompt Engineering** — Dynamic chain-of-thought templates with few-shot examples; query rewriting and HyDE (Hypothetical Document Embeddings)
* **RAGAS Evaluation** — Four core metrics tracked per query: answer relevancy, faithfulness, context precision, context recall
* **LangSmith Tracing** — Full chain observability, latency breakdowns, cost tracking
* **React Dashboard** — Live metric charts, query explorer, document ingestion UI

\---

## Quick Start

### Prerequisites

* Python 3.11+
* Node.js 18+
* API keys: OpenAI (required), LangSmith (optional but recommended)

### 1\. Clone \& Install

```bash
git clone https://github.com/YOUR\_USERNAME/rag-eval-suite.git
cd rag-eval-suite

# Backend
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2\. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3\. Run

```bash
# Terminal 1 — Backend
cd backend \&\& source venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

\---

## Environment Variables

|Variable|Required|Description|
|-|-|-|
|`OPENAI\_API\_KEY`|✅|OpenAI API key|
|`LANGCHAIN\_API\_KEY`|☑️|LangSmith tracing (set to `disabled` to skip)|
|`LANGCHAIN\_PROJECT`|☑️|LangSmith project name|
|`ANTHROPIC\_API\_KEY`|☑️|Claude as alternative LLM|
|`CHROMA\_PERSIST\_DIR`|—|ChromaDB path (default: `./data/chroma`)|

\---

## Evaluation Results

|Metric|Baseline|Optimized|Delta|
|-|-|-|-|
|Answer Relevancy|0.68|0.88|**+29.4%**|
|Faithfulness|0.71|0.91|**+28.2%**|
|Context Precision|0.64|0.83|**+29.7%**|
|Context Recall|0.72|0.87|**+20.8%**|

\---

## Project Structure

```
rag-eval-suite/
├── backend/
│   ├── api/            # FastAPI routes
│   ├── core/           # RAG pipeline (retriever, generator, reranker)
│   ├── evaluation/     # RAGAS + LangSmith wrappers
│   └── utils/          # Embeddings, document loaders
├── frontend/           # React + Vite dashboard
├── tests/              # Pytest integration tests
├── scripts/            # Seed data, batch eval runner
├── docs/               # Architecture diagrams
└── .env.example
```

\---

## Running Evaluations

```bash
# Run RAGAS evaluation on test set
cd backend
python -m scripts.run\_evaluation --dataset data/eval\_dataset.json --output results/

# View results
cat results/ragas\_report.json
```

\---

## Deployment

See [docs/deployment.md](docs/deployment.md) for Docker, Railway, and Render instructions.

\---

## License

MIT

