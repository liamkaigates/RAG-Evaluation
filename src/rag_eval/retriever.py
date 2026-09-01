from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import pickle
import re
import shutil
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from rank_bm25 import BM25Okapi

from rag_eval.data import Document

COLLECTION_NAME = "internal_docs"
METADATA_FILE = "metadata.json"
LEGACY_METADATA_FILE = "metadata.pkl"
CHROMA_DIR = "chroma"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 256
CHROMA_ADD_BATCH_SIZE = 1000


class EmbeddingModel(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbeddingProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    id: str
    title: str
    text: str
    score: float
    rank: int
    sparse_score: float
    dense_score: float
    source_path: str = ""
    section: str = ""
    citation: str = ""


class OpenAITextEmbeddingModel:
    name = OPENAI_EMBEDDING_MODEL
    dimension = 1536

    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            embeddings.extend(self._embed_batch(texts[start : start + EMBEDDING_BATCH_SIZE]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(model=self.name, input=texts)
        except AuthenticationError as exc:
            raise EmbeddingProviderError(
                "OpenAI authentication failed. Check that OPENAI_API_KEY is set to a valid key."
            ) from exc
        except RateLimitError as exc:
            raise EmbeddingProviderError(
                "OpenAI rejected the embedding request because the account is rate-limited or out of quota. "
                "Check billing/quota, or run `make build-internal-local` for offline development."
            ) from exc
        except APIConnectionError as exc:
            raise EmbeddingProviderError(
                "Could not reach OpenAI to create embeddings. Check network access, or run `make build-internal-local`."
            ) from exc
        except APIStatusError as exc:
            raise EmbeddingProviderError(f"OpenAI embedding request failed: {exc.message}") from exc
        return [item.embedding for item in response.data]


class LocalHashEmbeddingModel:
    name = "local-hash-embedding"
    dimension = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            index, sign = _hash_token(token, self.dimension)
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@lru_cache(maxsize=65536)
def _hash_token(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimension, 1.0 if digest[4] % 2 == 0 else -1.0


@dataclass
class RagIndex:
    path: Path
    documents: list[Document]
    bm25: BM25Okapi
    embedding_model: EmbeddingModel
    sparse_weight: float = 0.45
    _collection: Any = field(default=None, init=False, repr=False, compare=False)
    _collection_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @property
    def dense_dimension(self) -> int:
        return self.embedding_model.dimension

    @property
    def retrieval_mode(self) -> str:
        return f"{self.embedding_model.name}_chromadb_bm25_hybrid"

    def search(self, query: str, top_k: int = 5, sparse_weight: float | None = None) -> list[SearchResult]:
        if not query.strip():
            return []

        sparse_scores = self.bm25.get_scores(tokenize(query))
        dense_scores_by_id = self._dense_scores(query, fetch_k=max(top_k * 4, 20))
        dense_scores = [dense_scores_by_id.get(document.id, 0.0) for document in self.documents]

        sparse_normalized = min_max_scale(list(sparse_scores))
        dense_normalized = min_max_scale(dense_scores)
        sparse_ratio = self.sparse_weight if sparse_weight is None else max(0.0, min(1.0, sparse_weight))
        fused_scores = [
            (sparse_ratio * sparse_value) + ((1.0 - sparse_ratio) * dense_value)
            for sparse_value, dense_value in zip(sparse_normalized, dense_normalized, strict=True)
        ]

        candidate_indexes = [
            index
            for index in range(len(self.documents))
            if fused_scores[index] > 0 or sparse_scores[index] > 0 or dense_scores[index] > 0
        ]
        ranked_indexes = heapq.nlargest(top_k, candidate_indexes, key=fused_scores.__getitem__)
        results: list[SearchResult] = []
        for index in ranked_indexes:
            document = self.documents[index]
            results.append(
                SearchResult(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=float(fused_scores[index]),
                    rank=len(results) + 1,
                    sparse_score=float(sparse_scores[index]),
                    dense_score=float(dense_scores[index]),
                    source_path=document.source_path,
                    section=document.section,
                    citation=document.citation_label,
                )
            )
        return results

    def collection(self):
        with self._collection_lock:
            if self._collection is None:
                self._collection = open_collection(self.path)
            return self._collection

    def _dense_scores(self, query: str, fetch_k: int) -> dict[str, float]:
        collection = self.collection()
        query_embedding = self.embedding_model.embed([query])[0]
        payload = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, len(self.documents)),
            include=["distances"],
        )
        ids = payload.get("ids", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        return {doc_id: max(0.0, 1.0 - float(distance)) for doc_id, distance in zip(ids, distances, strict=True)}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9@-]*", text.lower())


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return values
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def embedding_model_from_env(provider: str | None = None) -> EmbeddingModel:
    selected = (provider or os.getenv("RAG_EMBEDDING_PROVIDER") or "openai").lower()
    if selected == "local":
        return LocalHashEmbeddingModel()
    if selected != "openai":
        raise ValueError("RAG_EMBEDDING_PROVIDER must be 'openai' or 'local'.")
    if not os.getenv("OPENAI_API_KEY"):
        raise EmbeddingProviderError(
            "OPENAI_API_KEY is required for OpenAI text embeddings. "
            "Set OPENAI_API_KEY for production, or run `make build-internal-local` for offline development."
        )
    return OpenAITextEmbeddingModel()


def open_collection(path: Path):
    client = chromadb.PersistentClient(
        path=str(path / CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(documents: list[Document], embedding_provider: str | None = None, dense_dimension: int | None = None) -> RagIndex:
    if not documents:
        raise ValueError("Cannot build an index with no documents.")

    embedding_model = embedding_model_from_env(embedding_provider)
    bm25 = BM25Okapi([tokenize(document.text) for document in documents])
    return RagIndex(path=Path(), documents=documents, bm25=bm25, embedding_model=embedding_model)


def save_index(index: RagIndex, path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)

    index.path = path
    index._collection = None
    collection = index.collection()
    texts = [document.text for document in index.documents]
    embeddings = index.embedding_model.embed(texts)
    metadatas = [
        {
            "title": document.title,
            "source_path": document.source_path,
            "section": document.section,
            "chunk_index": document.chunk_index,
        }
        for document in index.documents
    ]
    ids = [document.id for document in index.documents]
    for start in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
        stop = start + CHROMA_ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:stop],
            documents=texts[start:stop],
            embeddings=embeddings[start:stop],
            metadatas=metadatas[start:stop],
        )
    with (path / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "documents": [document.__dict__ for document in index.documents],
                "bm25_tokens": [tokenize(document.text) for document in index.documents],
                "embedding_provider": index.embedding_model.name,
                "embedding_dimension": index.embedding_model.dimension,
                "sparse_weight": index.sparse_weight,
            },
            handle,
            ensure_ascii=True,
        )


def load_index(path: str | Path, embedding_provider: str | None = None) -> RagIndex:
    path = Path(path)
    metadata_path = path / METADATA_FILE
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        documents = [Document(**row) for row in metadata["documents"]]
        token_lists = metadata.get("bm25_tokens") or [tokenize(document.text) for document in documents]
        bm25 = BM25Okapi(token_lists)
    else:
        # Legacy pre-JSON index artifacts; only load index directories you built yourself.
        with (path / LEGACY_METADATA_FILE).open("rb") as handle:
            metadata = pickle.load(handle)
        documents = metadata["documents"]
        bm25 = metadata.get("bm25") or BM25Okapi([tokenize(document.text) for document in documents])

    stored_provider = metadata.get("embedding_provider", "")
    selected_provider = embedding_provider
    if selected_provider is None and stored_provider == LocalHashEmbeddingModel.name:
        selected_provider = "local"
    embedding_model = embedding_model_from_env(selected_provider)
    if embedding_model.dimension != metadata.get("embedding_dimension"):
        raise ValueError("Embedding model dimension does not match the stored ChromaDB index.")
    return RagIndex(
        path=path,
        documents=documents,
        bm25=bm25,
        embedding_model=embedding_model,
        sparse_weight=float(metadata.get("sparse_weight", 0.45)),
    )
