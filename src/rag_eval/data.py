from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_path: str = ""
    section: str = ""
    chunk_index: int = 0

    @property
    def citation_label(self) -> str:
        if self.section:
            return f"{self.title}, {self.section}"
        return self.title


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    relevant_doc_ids: list[str]
    answer_keywords: list[str]


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_documents(path: str | Path) -> list[Document]:
    documents = []
    for row in read_jsonl(path):
        documents.append(
            Document(
                id=row["id"],
                title=row["title"],
                text=row["text"],
                source_path=row.get("source_path", ""),
                section=row.get("section", ""),
                chunk_index=int(row.get("chunk_index", 0)),
            )
        )
    return documents


def load_questions(path: str | Path) -> list[EvalQuestion]:
    return [
        EvalQuestion(
            id=row["id"],
            question=row["question"],
            relevant_doc_ids=list(row["relevant_doc_ids"]),
            answer_keywords=list(row["answer_keywords"]),
        )
        for row in read_jsonl(path)
    ]
