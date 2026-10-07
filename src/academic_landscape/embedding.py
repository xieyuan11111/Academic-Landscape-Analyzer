"""Thin Qdrant adapter for versioned, source-bound embeddings."""

from __future__ import annotations

import pickle
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.local.qdrant_local import QdrantLocal

from .chunking import Chunk
from .models import CorpusSnapshot


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int | None

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class OpenAIEmbeddingProvider:
    """Thin adapter around the official OpenAI-compatible embeddings SDK."""

    def __init__(
        self,
        model_name: str,
        *,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self._client = client
        self._dimension = dimension

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model_name,
            input=list(texts),
            encoding_format="float",
        )
        items = sorted(response.data, key=lambda item: int(item.index))
        if len(items) != len(texts):
            raise ValueError("embedding provider returned an unexpected item count")
        vectors = [tuple(float(value) for value in item.embedding) for item in items]
        dimension = len(vectors[0]) if vectors else 0
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding provider returned inconsistent dimensions")
        if self._dimension is not None and self._dimension != dimension:
            raise ValueError(
                f"embedding dimension {dimension} does not match configured dimension {self._dimension}"
            )
        self._dimension = dimension
        return vectors


def embed_chunks(
    provider: EmbeddingProvider,
    chunks: Sequence[Chunk],
    *,
    batch_size: int = 64,
) -> list["EmbeddingVector"]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    vectors: list[EmbeddingVector] = []
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        batch_vectors = provider.embed([chunk.text for chunk in batch])
        if len(batch_vectors) != len(batch):
            raise ValueError("embedding provider returned an unexpected item count")
        vectors.extend(
            EmbeddingVector(chunk_id=chunk.chunk_id, values=values)
            for chunk, values in zip(batch, batch_vectors, strict=True)
        )
    return vectors


@dataclass(frozen=True)
class EmbeddingVector:
    chunk_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class QdrantHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class QdrantEmbeddingStore:
    """Store and query embeddings without making Qdrant the evidence authority."""

    def __init__(
        self,
        url: str,
        collection_prefix: str,
        model_name: str,
        dimension: int,
        *,
        embedding_version: str = "v1",
        timeout: float = 120.0,
        client: QdrantClient | None = None,
        local_path: str | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not collection_prefix.strip():
            raise ValueError("collection_prefix must not be empty")
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.url = url
        self.collection_prefix = collection_prefix
        self.model_name = model_name
        self.dimension = dimension
        self.embedding_version = embedding_version
        self.timeout = timeout
        self.local_path = local_path
        self._owns_client = client is None
        self._client = client or (
            QdrantLocal(local_path) if local_path else QdrantClient(url=url, timeout=timeout)
        )
        self.collection_name = self._collection_name()

    def _collection_name(self) -> str:
        def safe(value: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()

        return (
            f"{safe(self.collection_prefix)}__chunks__{safe(self.model_name)}__"
            f"{self.dimension}d__{safe(self.embedding_version)}"
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"academic-landscape:{chunk_id}"))

    @staticmethod
    def _snapshot_filter(snapshot_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="snapshot_id",
                    match=models.MatchValue(value=snapshot_id),
                )
            ]
        )

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def recreate_collection(self, snapshot_id: str) -> None:
        """Reset only this snapshot's points, keeping other snapshots intact."""
        self._ensure_collection()
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self._snapshot_filter(snapshot_id),
            wait=True,
        )

    def upsert(
        self,
        snapshot: CorpusSnapshot,
        chunks: Sequence[Chunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        self._ensure_collection()
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.snapshot_id != snapshot.snapshot_id:
                raise ValueError("chunk snapshot_id does not match snapshot")
            if vector.chunk_id != chunk.chunk_id:
                raise ValueError("embedding chunk_id does not match chunk")
            if len(vector.values) != self.dimension:
                raise ValueError(
                    f"embedding dimension {len(vector.values)} does not match configured dimension {self.dimension}"
                )
            payload = chunk.payload()
            payload.update(
                {
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.dimension,
                    "embedding_version": self.embedding_version,
                }
            )
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(vector.values),
                    payload=payload,
                )
            )
        if points:
            if self.local_path:
                self._ensure_collection()
                self._upsert_local_batch(points)
            else:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

    def _upsert_local_batch(
        self,
        points: Sequence[models.PointStruct],
    ) -> None:
        """Run Qdrant's in-memory upsert path with one SQLite commit."""
        local_collection = self._client.collections[self.collection_name]
        persistence = local_collection.storage

        def persist_without_commit(point: models.PointStruct) -> None:
            cursor = persistence.storage.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO points VALUES (?, ?)",
                (
                    type(persistence).encode_key(point.id),
                    sqlite3.Binary(pickle.dumps(point)),
                ),
            )

        original_persist = persistence.persist
        persistence.persist = persist_without_commit
        try:
            local_collection.upsert(points)
            persistence.storage.commit()
        except Exception:
            persistence.storage.rollback()
            raise
        finally:
            persistence.persist = original_persist

    def search(
        self,
        snapshot_id: str,
        query_vector: Sequence[float],
        *,
        limit: int = 10,
    ) -> list[QdrantHit]:
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match configured dimension {self.dimension}"
            )
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._ensure_collection()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=self._snapshot_filter(snapshot_id),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            QdrantHit(
                chunk_id=str(point.payload.get("chunk_id")),
                score=float(point.score),
                payload=dict(point.payload),
            )
            for point in response.points
        ]

    def delete_collection(self, snapshot_id: str | None = None) -> None:
        """Delete the whole project collection, optionally checking snapshot scope first."""
        if not self._client.collection_exists(self.collection_name):
            return
        if snapshot_id is not None:
            # Keep the argument for call-site compatibility, but never silently
            # delete a different collection: callers use this only for this store.
            self._client.delete_collection(collection_name=self.collection_name)
        else:
            self._client.delete_collection(collection_name=self.collection_name)


    def count(self, snapshot_id: str | None = None) -> int:
        if not self._client.collection_exists(self.collection_name):
            return 0
        result = self._client.count(
            collection_name=self.collection_name,
            count_filter=None if snapshot_id is None else self._snapshot_filter(snapshot_id),
            exact=True,
        )
        return int(result.count)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QdrantEmbeddingStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
