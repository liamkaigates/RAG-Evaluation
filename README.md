# Hybrid RAG Platform

Production-shaped retrieval-augmented generation project for ingesting internal company documentation, indexing it with hybrid retrieval, and generating grounded answers with source citations.

The platform ingests Markdown/text documentation, chunks it with source metadata, stores OpenAI text embeddings in ChromaDB, indexes sparse keywords with BM25, retrieves relevant context for employee questions, and generates GPT-4o answers with inline citations like `[S1]`.

## What It Demonstrates

- Internal documentation ingestion with section-aware chunking and source-path metadata.
- LangChain `RecursiveCharacterTextSplitter` chunking.
- Dense retrieval with OpenAI `text-embedding-3-small` embeddings stored in ChromaDB.
- Sparse retrieval with BM25 via `rank_bm25`.
- Score fusion with per-result sparse, dense, and hybrid score diagnostics.
- GPT-4o grounded answer generation with inline citations and structured source objects.
- Retrieval metrics: Recall@K, Precision@K, Hit Rate@K, MRR, and NDCG.
- Answer quality metrics: keyword coverage, citation support, and context faithfulness proxy.
- FastAPI dashboard/API with ranked passages, citations, answer text, and evaluation summaries.
- Python 3.11+, reproducible CLI, tests, and Docker deployment.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...
make build-internal
make query-internal
make serve-internal
```

Or use the project-managed local environment:

```bash
make setup
export OPENAI_API_KEY=...
make build-internal
```

Open:

```text
http://127.0.0.1:8080
```

## Commands

Production commands use OpenAI embeddings and GPT-4o:

```bash
make ingest
make build-index
make build-internal
make query
make query-internal
make evaluate
make test
make serve
make serve-internal
```

Offline/local commands use deterministic local embeddings and extractive generation for tests:

```bash
make build-internal-local
make query-internal-local
make evaluate-local
make test
make serve-internal-local
```

## Environment

```bash
OPENAI_API_KEY=...
RAG_EMBEDDING_PROVIDER=openai  # default; use local for tests
RAG_GENERATION_PROVIDER=openai # default; use local for tests
```

The production path uses OpenAI `text-embedding-3-small` for document/query embeddings and `gpt-4o` for grounded answer generation.
`make build-internal` requires an OpenAI API key with available embedding quota. If the API returns `insufficient_quota`, use `make build-internal-local` while developing locally, or enable billing/quota on the OpenAI account before rebuilding the production index.

## Data Format

Corpus JSONL:

```json
{"id": "doc_id", "title": "Document title", "text": "Document text..."}
```

Evaluation JSONL:

```json
{"id": "q1", "question": "What is retrieval?", "relevant_doc_ids": ["doc_id"], "answer_keywords": ["retrieval", "context"]}
```

Internal docs can be placed under `docs/internal` as `.md` or `.txt` files. `make ingest` writes chunked JSONL to `data/internal_corpus.jsonl`, and `make build-internal` builds the ChromaDB/BM25 artifact directory at `artifacts/internal_index`.

## Docker

```bash
docker build -t hybrid-rag-platform .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY hybrid-rag-platform
```

The container ingests `docs/internal`, builds the ChromaDB/BM25 hybrid index at startup, and serves the FastAPI dashboard on port `8080`.
