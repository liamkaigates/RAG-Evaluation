from __future__ import annotations

import argparse
import json

from rag_eval.data import load_documents, load_questions
from rag_eval.evaluate import evaluate_questions
from rag_eval.generator import generate_answer
from rag_eval.ingest import ingest_directory, write_corpus
from rag_eval.retriever import build_index, load_index, save_index


def ingest_command(args: argparse.Namespace) -> None:
    documents = ingest_directory(args.source_dir, max_words=args.chunk_words, overlap=args.chunk_overlap)
    write_corpus(documents, args.output)
    print(
        json.dumps(
            {
                "status": "ingested",
                "source_dir": args.source_dir,
                "chunks": len(documents),
                "output": args.output,
            },
            indent=2,
        )
    )


def build_index_command(args: argparse.Namespace) -> None:
    index = build_index(load_documents(args.corpus), embedding_provider=args.embedding_provider)
    save_index(index, args.index)
    print(
        json.dumps(
            {
                "status": "indexed",
                "chunks": len(index.documents),
                "dense_dimension": index.dense_dimension,
                "retrieval": index.retrieval_mode,
                "index": args.index,
            },
            indent=2,
        )
    )


def query_command(args: argparse.Namespace) -> None:
    index = load_index(args.index, embedding_provider=args.embedding_provider)
    results = index.search(args.question, top_k=args.top_k)
    generated = generate_answer(args.question, results, provider=args.generation_provider)
    print(
        json.dumps(
            {
                "question": args.question,
                "answer": generated["answer"],
                "citations": generated["citations"],
                "sources": generated["sources"],
                "model": generated["model"],
                "results": [result.__dict__ for result in results],
            },
            indent=2,
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    index = load_index(args.index, embedding_provider=args.embedding_provider)
    questions = load_questions(args.questions)
    print(json.dumps(evaluate_questions(index, questions, top_k=args.top_k, generation_provider=args.generation_provider), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--chunk-words", type=int, default=180)
    ingest.add_argument("--chunk-overlap", type=int, default=35)
    ingest.set_defaults(func=ingest_command)

    build = subparsers.add_parser("build-index")
    build.add_argument("--corpus", required=True)
    build.add_argument("--index", required=True)
    build.add_argument("--embedding-provider", choices=["openai", "local"], default=None)
    build.set_defaults(func=build_index_command)

    query = subparsers.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--embedding-provider", choices=["openai", "local"], default=None)
    query.add_argument("--generation-provider", choices=["openai", "local"], default=None)
    query.set_defaults(func=query_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--embedding-provider", choices=["openai", "local"], default=None)
    evaluate.add_argument("--generation-provider", choices=["openai", "local"], default=None)
    evaluate.set_defaults(func=evaluate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
