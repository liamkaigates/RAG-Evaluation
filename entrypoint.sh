#!/bin/sh
set -e

INDEX_DIR="artifacts/internal_index"

if [ -f "$INDEX_DIR/metadata.json" ] || [ -f "$INDEX_DIR/metadata.pkl" ]; then
    echo "Reusing existing index at $INDEX_DIR (mount /app/artifacts as a volume to persist it)."
else
    echo "No index found at $INDEX_DIR; ingesting and building..."
    python -m rag_eval.cli ingest --source-dir docs/internal --output data/internal_corpus.jsonl
    python -m rag_eval.cli build-index --corpus data/internal_corpus.jsonl --index "$INDEX_DIR"
fi

exec python -m rag_eval.api --index "$INDEX_DIR" --questions data/internal_eval_questions.jsonl --host 0.0.0.0 --port 8080
