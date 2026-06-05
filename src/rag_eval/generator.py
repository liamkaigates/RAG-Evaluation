from __future__ import annotations

import os
import re
from typing import List

from openai import OpenAI

from rag_eval.retriever import SearchResult


GPT_MODEL = "gpt-4o"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9@-]*", text.lower()))


def split_sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def generate_answer(question: str, results: List[SearchResult], max_sentences: int = 3, provider: str | None = None) -> dict:
    selected_provider = (provider or os.getenv("RAG_GENERATION_PROVIDER") or "openai").lower()
    if selected_provider == "local":
        return generate_extractive_answer(question, results, max_sentences=max_sentences)
    if selected_provider != "openai":
        raise ValueError("RAG_GENERATION_PROVIDER must be 'openai' or 'local'.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for GPT-4o answer generation. Use RAG_GENERATION_PROVIDER=local for offline tests.")
    return generate_gpt4o_answer(question, results, max_sentences=max_sentences)


def generate_gpt4o_answer(question: str, results: List[SearchResult], max_sentences: int = 3) -> dict:
    if not results:
        return empty_answer()

    citation_numbers = {result.id: index + 1 for index, result in enumerate(results)}
    context = "\n\n".join(
        f"[S{citation_numbers[result.id]}] {result.citation}\n{result.text}"
        for result in results
    )
    client = OpenAI()
    response = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided company documentation context. "
                    "Every factual claim must include an inline source citation like [S1]. "
                    "If the context does not support an answer, say that the documentation does not contain enough information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Write a concise answer in at most {max_sentences} sentences."
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    cited_ids = [result.id for result in results if f"[S{citation_numbers[result.id]}]" in content]
    return {
        "answer": content or "No supported answer found in retrieved context.",
        "citations": cited_ids,
        "sources": source_payload([result for result in results if result.id in cited_ids], citation_numbers),
        "model": GPT_MODEL,
    }


def generate_extractive_answer(question: str, results: List[SearchResult], max_sentences: int = 3) -> dict:
    query_terms = tokenize(question)
    candidates = []
    for result in results:
        for sentence in split_sentences(result.text):
            overlap = len(query_terms.intersection(tokenize(sentence)))
            candidates.append((overlap, result.score, result, sentence))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_sentences = []
    selected_results = []
    seen_sentences = set()
    for _, _, result, sentence in candidates:
        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            selected_sentences.append(sentence)
            selected_results.append(result)
        if len(selected_sentences) >= max_sentences:
            break

    cited_ids: list[str] = []
    citation_numbers: dict[str, int] = {}
    answer_parts: list[str] = []
    for sentence, result in zip(selected_sentences, selected_results):
        if result.id not in citation_numbers:
            citation_numbers[result.id] = len(citation_numbers) + 1
            cited_ids.append(result.id)
        answer_parts.append(f"{sentence} [S{citation_numbers[result.id]}]")

    return {
        "answer": " ".join(answer_parts) if answer_parts else "No supported answer found in retrieved context.",
        "citations": cited_ids,
        "sources": source_payload(selected_results, citation_numbers),
        "model": "local-extractive",
    }


def empty_answer() -> dict:
    return {
        "answer": "No supported answer found in retrieved context.",
        "citations": [],
        "sources": [],
        "model": "none",
    }


def source_payload(results: list[SearchResult], citation_numbers: dict[str, int]) -> list[dict]:
    sources = []
    for result in results:
        if result.id not in citation_numbers:
            continue
        sources.append(
            {
                "id": result.id,
                "label": f"S{citation_numbers[result.id]}",
                "title": result.title,
                "source_path": result.source_path,
                "section": result.section,
                "citation": result.citation,
            }
        )
    return dedupe_sources(sources)


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for source in sources:
        if source["id"] in seen:
            continue
        seen.add(source["id"])
        unique.append(source)
    return unique
