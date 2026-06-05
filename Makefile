.PHONY: ingest build-index build-internal query query-internal evaluate test serve serve-internal

PYTHONPATH := src
PYTHON ?= python3.11
CORPUS := data/corpus.jsonl
QUESTIONS := data/eval_questions.jsonl
INDEX := artifacts/index
INTERNAL_DOCS := docs/internal
INTERNAL_CORPUS := data/internal_corpus.jsonl
INTERNAL_INDEX := artifacts/internal_index
INTERNAL_QUESTIONS := data/internal_eval_questions.jsonl

ingest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli ingest --source-dir $(INTERNAL_DOCS) --output $(INTERNAL_CORPUS)

build-index:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli build-index --corpus $(CORPUS) --index $(INDEX)

build-internal: ingest
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli build-index --corpus $(INTERNAL_CORPUS) --index $(INTERNAL_INDEX)

query:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli query --index $(INDEX) --question "How does RAG reduce hallucination?" --top-k 4

query-internal:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli query --index $(INTERNAL_INDEX) --question "How should production RAG answers cite sources?" --top-k 4

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli evaluate --index $(INDEX) --questions $(QUESTIONS) --top-k 4

test:
	RAG_EMBEDDING_PROVIDER=local RAG_GENERATION_PROVIDER=local PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

serve:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.api --index $(INDEX) --questions $(QUESTIONS) --port 8080

serve-internal:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.api --index $(INTERNAL_INDEX) --questions $(INTERNAL_QUESTIONS) --port 8080

.PHONY: build-internal-local query-internal-local evaluate-local serve-internal-local

build-internal-local: ingest
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli build-index --corpus $(INTERNAL_CORPUS) --index $(INTERNAL_INDEX) --embedding-provider local

query-internal-local:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli query --index $(INTERNAL_INDEX) --question "How should production RAG answers cite sources?" --top-k 4 --embedding-provider local --generation-provider local

evaluate-local:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.cli evaluate --index $(INDEX) --questions $(QUESTIONS) --top-k 4 --embedding-provider local --generation-provider local

serve-internal-local:
	RAG_EMBEDDING_PROVIDER=local RAG_GENERATION_PROVIDER=local PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m rag_eval.api --index $(INTERNAL_INDEX) --questions $(INTERNAL_QUESTIONS) --port 8080
