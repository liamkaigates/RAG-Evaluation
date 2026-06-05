from __future__ import annotations

from typing import Dict, List

from rag_eval.data import EvalQuestion
from rag_eval.generator import generate_answer
from rag_eval.metrics import answer_metrics, mean_metrics, retrieval_metrics
from rag_eval.retriever import RagIndex


def evaluate_questions(index: RagIndex, questions: List[EvalQuestion], top_k: int = 5, generation_provider: str | None = None) -> dict:
    rows = []
    retrieval_rows = []
    answer_rows = []
    for question in questions:
        results = index.search(question.question, top_k=top_k)
        generated = generate_answer(question.question, results, provider=generation_provider)
        retrieval = retrieval_metrics(question, results)
        answer = answer_metrics(question, generated["answer"], generated["citations"], results)
        retrieval_rows.append(retrieval)
        answer_rows.append(answer)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "answer": generated["answer"],
                "citations": generated["citations"],
                "sources": generated["sources"],
                "retrieved_doc_ids": [result.id for result in results],
                "retrieval": retrieval,
                "answer_metrics": answer,
            }
        )
    return {
        "summary": {
            **mean_metrics(retrieval_rows),
            **mean_metrics(answer_rows),
            "question_count": float(len(questions)),
            "top_k": float(top_k),
        },
        "questions": rows,
    }
