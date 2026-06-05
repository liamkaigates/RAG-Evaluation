from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.data import Document


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_markdown_sections(text: str) -> List[tuple[str, str]]:
    sections: List[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(heading, body) for heading, body in sections if body]


def ingest_directory(source_dir: str | Path, max_words: int = 180, overlap: int = 35) -> List[Document]:
    source_root = Path(source_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Documentation directory not found: {source_root}")

    chunk_size = max(200, max_words * 6)
    chunk_overlap = max(0, min(overlap * 6, chunk_size // 2))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = normalize_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        title = path.stem.replace("_", " ").replace("-", " ").title()
        relative_path = str(path.relative_to(source_root))
        doc_slug = slugify(relative_path.removesuffix(path.suffix))
        chunk_index = 0
        for section, body in split_markdown_sections(text):
            for chunk in splitter.split_text(body):
                documents.append(
                    Document(
                        id=f"{doc_slug}#chunk-{chunk_index:03d}",
                        title=title,
                        text=chunk,
                        source_path=relative_path,
                        section=section,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
    return documents


def write_corpus(documents: List[Document], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "id": document.id,
                        "title": document.title,
                        "text": document.text,
                        "source_path": document.source_path,
                        "section": document.section,
                        "chunk_index": document.chunk_index,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
