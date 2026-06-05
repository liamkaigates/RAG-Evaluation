FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY docs ./docs
COPY src ./src
COPY static ./static
COPY tests ./tests

ENV PYTHONPATH=src

EXPOSE 8080
CMD ["sh", "-c", "python -m rag_eval.cli ingest --source-dir docs/internal --output data/internal_corpus.jsonl && python -m rag_eval.cli build-index --corpus data/internal_corpus.jsonl --index artifacts/internal_index && python -m rag_eval.api --index artifacts/internal_index --questions data/internal_eval_questions.jsonl --host 0.0.0.0 --port 8080"]
