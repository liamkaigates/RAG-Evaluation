from __future__ import annotations

import math
from typing import Dict, List

from rag_eval.data import EvalQuestion
from rag_eval.generator import tokenize
from rag_eval.retriever import SearchResult


def dcg(relevance: List[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))


def retrieval_metrics(question: EvalQuestion, results: List[SearchResult]) -> Dict[str, float]:
    relevant = set(question.relevant_doc_ids)
    retrieved = [result.id for result in results]
    hits = [1 if doc_id in relevant else 0 for doc_id in retrieved]
    hit_count = sum(hits)
    ideal = sorted(hits, reverse=True)
    reciprocal = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            reciprocal = 1.0 / index
            break
    return {
        "recall_at_k": hit_count / len(relevant) if relevant else 0.0,
        "precision_at_k": hit_count / len(retrieved) if retrieved else 0.0,
        "hit_rate_at_k": 1.0 if hit_count else 0.0,
        "mrr": reciprocal,
        "ndcg": dcg(hits) / dcg(ideal) if any(ideal) else 0.0,
    }


def answer_metrics(question: EvalQuestion, answer: str, citations: List[str], context: List[SearchResult]) -> Dict[str, float]:
    answer_terms = tokenize(answer)
    keywords = [keyword.lower() for keyword in question.answer_keywords]
    keyword_hits = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    context_text = " ".join(result.text for result in context).lower()
    answer_tokens = tokenize(answer)
    supported_tokens = [token for token in answer_tokens if token in context_text]
    relevant_citations = set(citations).intersection(question.relevant_doc_ids)
    return {
        "keyword_coverage": keyword_hits / len(keywords) if keywords else 0.0,
        "citation_support": len(relevant_citations) / len(set(citations)) if citations else 0.0,
        "faithfulness_proxy": len(supported_tokens) / len(answer_tokens) if answer_tokens else 0.0,
    }


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}

