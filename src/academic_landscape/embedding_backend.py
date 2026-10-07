"""Embedding backends for semantic clustering and thread analysis."""

from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to cuda when usable, else cpu.

    A hard ``device="cuda"`` default made a CPU-only install fail with
    ``Torch not compiled with CUDA enabled`` before doing any work — the first
    thing a new user would see. An explicitly named device is returned verbatim,
    because a caller who names one means it and should get a loud failure rather
    than a silent downgrade.
    """
    if device != "auto":
        return device
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch absent or broken: cpu is the answer
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a broken cuda probe is not a reason to crash
        return "cpu"


class SiliconFlowEmbeddingBackend:
    """OpenAI-compatible SiliconFlow embeddings with strict response validation.

    This is deliberately an API backend rather than a hidden local fallback:
    a configured-but-failing provider must fail loudly so a report never mixes
    embedding models or silently bypasses the user's configured route.
    """

    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    DEFAULT_MODEL = "BAAI/bge-m3"
    MAX_INPUT_CHARS = 50_000

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        batch_size: int = 16,
        max_input_chars: int = MAX_INPUT_CHARS,
        client: object | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SiliconFlow API key must not be empty")
        if not model_name.strip():
            raise ValueError("SiliconFlow embedding model must not be empty")
        if not base_url.strip():
            raise ValueError("SiliconFlow base URL must not be empty")
        if timeout <= 0:
            raise ValueError("SiliconFlow timeout must be positive")
        if batch_size <= 0:
            raise ValueError("SiliconFlow batch_size must be positive")
        if max_input_chars <= 0:
            raise ValueError("SiliconFlow max_input_chars must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_input_chars = max_input_chars
        self._client = client
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        """Dimension learned from the first validated provider response."""
        return self._dimension

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Call the provider once and validate its batch-local response order."""
        for index, text in enumerate(texts):
            if len(text) > self.max_input_chars:
                raise ValueError(
                    f"SiliconFlow input item at index {index} has {len(text)} characters; "
                    f"maximum allowed is {self.max_input_chars}"
                )

        response = self._client.embeddings.create(
            model=self.model_name,
            input=list(texts),
            encoding_format="float",
        )
        raw_items = list(response.data)
        if len(raw_items) != len(texts):
            raise ValueError("SiliconFlow returned an unexpected embedding item count")
        try:
            indexed = [(int(item.index), item.embedding) for item in raw_items]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("SiliconFlow returned malformed embedding items") from exc
        indexed.sort(key=lambda item: item[0])
        if [index for index, _ in indexed] != list(range(len(texts))):
            raise ValueError("SiliconFlow returned missing or duplicate embedding indices")
        try:
            vectors = np.asarray([embedding for _, embedding in indexed], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("SiliconFlow returned non-numeric embedding values") from exc
        if vectors.ndim != 2 or vectors.shape[1] <= 0 or not np.isfinite(vectors).all():
            raise ValueError("SiliconFlow returned invalid embedding vectors")
        return vectors

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed input in bounded requests, preserving original input order.

        A whole corpus can contain thousands of chunks.  Sending all of them in
        one OpenAI-compatible request causes provider-side request timeouts, so
        batching happens in the provider adapter—not only at Qdrant upsert.
        """
        if not texts:
            return np.zeros((0, 0), dtype=float)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batches.append(self._encode_batch(texts[start : start + self.batch_size]))
        vectors = np.vstack(batches)
        dimension = int(vectors.shape[1])
        if self._dimension is not None and self._dimension != dimension:
            raise ValueError(
                f"SiliconFlow embedding dimension {dimension} does not match prior dimension "
                f"{self._dimension}"
            )
        self._dimension = dimension
        return vectors


class LocalEmbeddingBackend:
    """SentenceTransformer backend; uses cached weights offline by default."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        offline: bool = True,
    ) -> None:
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
        )
