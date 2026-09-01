from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from rag_eval.data import EvalQuestion
from rag_eval.generator import generate_answer
from rag_eval.metrics import answer_metrics, mean_metrics, retrieval_metrics
from rag_eval.retriever import RagIndex


def evaluate_questions(
    index: RagIndex,
    questions: List[EvalQuestion],
    top_k: int = 5,
    generation_provider: str | None = None,
    max_workers: int = 8,
) -> dict:
    def evaluate_one(question: EvalQuestion) -> tuple[dict, Dict[str, float], Dict[str, float]]:
        results = index.search(question.question, top_k=top_k)
        generated = generate_answer(question.question, results, provider=generation_provider)
        retrieval = retrieval_metrics(question, results)
        answer = answer_metrics(question, generated["answer"], generated["citations"], results)
        row = {
            "id": question.id,
            "question": question.question,
            "answer": generated["answer"],
            "citations": generated["citations"],
            "sources": generated["sources"],
            "retrieved_doc_ids": [result.id for result in results],
            "retrieval": retrieval,
            "answer_metrics": answer,
        }
        return row, retrieval, answer

    if len(questions) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(questions))) as executor:
            evaluated = list(executor.map(evaluate_one, questions))
    else:
        evaluated = [evaluate_one(question) for question in questions]

    rows = [row for row, _, _ in evaluated]
    retrieval_rows = [retrieval for _, retrieval, _ in evaluated]
    answer_rows = [answer for _, _, answer in evaluated]
    return {
        "summary": {
            **mean_metrics(retrieval_rows),
            **mean_metrics(answer_rows),
            "question_count": float(len(questions)),
            "top_k": float(top_k),
        },
        "questions": rows,
    }
