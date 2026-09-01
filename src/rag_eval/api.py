from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from rag_eval.data import EvalQuestion, load_questions
from rag_eval.evaluate import evaluate_questions
from rag_eval.generator import generate_answer
from rag_eval.retriever import EmbeddingProviderError, RagIndex, load_index

PROVIDER_ERRORS = (EmbeddingProviderError, OpenAIError, RuntimeError)


def create_app(index: RagIndex, questions: list[EvalQuestion], static_dir: str | Path = "static") -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG Platform",
        description="Company-documentation RAG API using OpenAI embeddings, ChromaDB, BM25, GPT-4o, and FastAPI.",
        version="0.1.0",
    )
    static_path = Path(static_dir)
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    source_count = len({document.source_path or document.id for document in index.documents})
    summary_payload = {
        "documents": source_count,
        "chunks": len(index.documents),
        "questions": len(questions),
        "dense_dimension": index.dense_dimension,
        "retrieval": index.retrieval_mode,
        "generation_model": "gpt-4o" if os.getenv("RAG_GENERATION_PROVIDER", "openai").lower() == "openai" else "local-extractive",
    }
    eval_cache: dict[int, dict] = {}
    eval_cache_lock = threading.Lock()

    @app.get("/")
    @app.get("/dashboard")
    def dashboard() -> FileResponse:
        return FileResponse(static_path / "index.html")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "chunks": len(index.documents)}

    @app.get("/api/summary")
    def summary() -> dict:
        return summary_payload

    @app.get("/api/search")
    def search(question: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)) -> dict:
        try:
            results = index.search(question, top_k=top_k)
            generated = generate_answer(question, results)
        except PROVIDER_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "question": question,
            "answer": generated["answer"],
            "citations": generated["citations"],
            "sources": generated["sources"],
            "model": generated["model"],
            "results": [result.__dict__ for result in results],
        }

    @app.get("/api/evaluate")
    def evaluate(top_k: int = Query(5, ge=1, le=20), refresh: bool = Query(False)) -> dict:
        if not refresh:
            with eval_cache_lock:
                cached = eval_cache.get(top_k)
            if cached is not None:
                return cached
        try:
            payload = evaluate_questions(index, questions, top_k=top_k)
        except PROVIDER_ERRORS as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        with eval_cache_lock:
            eval_cache[top_k] = payload
        return payload

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the FastAPI RAG dashboard")
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app = create_app(load_index(args.index), load_questions(args.questions), args.static_dir)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
