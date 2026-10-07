"""SiliconFlow must be the CLI's explicit API-default embedding route."""

from __future__ import annotations

import argparse
import types

import numpy as np
import pytest

import academic_landscape.cli as cli
from academic_landscape.embedding_backend import SiliconFlowEmbeddingBackend


class _FakeEmbeddings:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.embeddings = _FakeEmbeddings(response)


def _embedding(index: int, values: list[float]) -> object:
    return types.SimpleNamespace(index=index, embedding=values)


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "embedding_backend": "siliconflow",
        "embedding_api_key_env": "TEST_SILICONFLOW_KEY",
        "embedding_base_url": "https://api.siliconflow.cn/v1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_timeout": 30.0,
        "local_embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_device": "auto",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_siliconflow_backend_restores_original_input_order() -> None:
    response = types.SimpleNamespace(
        data=[_embedding(1, [3.0, 4.0]), _embedding(0, [1.0, 2.0])]
    )
    client = _FakeClient(response)
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=client)

    vectors = backend.encode(["first", "second"])

    assert isinstance(vectors, np.ndarray)
    assert vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert backend.dimension == 2
    assert client.embeddings.calls == [
        {"model": "BAAI/bge-m3", "input": ["first", "second"], "encoding_format": "float"}
    ]



class _BatchingEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        return types.SimpleNamespace(
            data=[_embedding(index, [float(len(text)), float(index)]) for index, text in enumerate(texts)]
        )


def test_siliconflow_backend_batches_large_input_without_reordering() -> None:
    client = types.SimpleNamespace(embeddings=_BatchingEmbeddings())
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=client, batch_size=2)

    vectors = backend.encode(["a", "bb", "ccc", "dddd", "eeeee"])

    assert vectors.tolist() == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    assert [call["input"] for call in client.embeddings.calls] == [
        ["a", "bb"], ["ccc", "dddd"], ["eeeee"]
    ]


def test_siliconflow_backend_accepts_input_at_50000_character_limit() -> None:
    response = types.SimpleNamespace(data=[_embedding(0, [1.0, 2.0])])
    client = _FakeClient(response)
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=client)
    text = "a" * 50_000

    vectors = backend.encode([text])

    assert vectors.shape == (1, 2)
    assert client.embeddings.calls == [
        {"model": "BAAI/bge-m3", "input": [text], "encoding_format": "float"}
    ]


def test_siliconflow_backend_rejects_input_over_50000_characters() -> None:
    client = _FakeClient(types.SimpleNamespace(data=[]))
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=client)

    with pytest.raises(ValueError, match="50000"):
        backend.encode(["a" * 50_001])

    assert client.embeddings.calls == []


def test_siliconflow_backend_rejects_nonpositive_max_input_chars() -> None:
    with pytest.raises(ValueError, match="max_input_chars"):
        SiliconFlowEmbeddingBackend(api_key="test-key", max_input_chars=0)


def test_siliconflow_backend_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SiliconFlowEmbeddingBackend(api_key="test-key", batch_size=0)


def test_siliconflow_backend_rejects_missing_or_duplicate_response_indices() -> None:
    response = types.SimpleNamespace(
        data=[_embedding(0, [1.0, 2.0]), _embedding(0, [3.0, 4.0])]
    )
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=_FakeClient(response))

    with pytest.raises(ValueError, match="indices"):
        backend.encode(["first", "second"])


def test_siliconflow_backend_rejects_dimension_drift() -> None:
    client = _FakeClient(types.SimpleNamespace(data=[_embedding(0, [1.0, 2.0])]))
    backend = SiliconFlowEmbeddingBackend(api_key="test-key", client=client)
    assert backend.encode(["first"]).shape == (1, 2)
    client.embeddings.response = types.SimpleNamespace(data=[_embedding(0, [1.0, 2.0, 3.0])])

    with pytest.raises(ValueError, match="dimension"):
        backend.encode(["second"])


def test_landscape_parser_defaults_to_siliconflow_api() -> None:
    args = cli._parser().parse_args(["landscape", "snapshot", "--output", "out"])

    assert args.embedding_backend == "siliconflow"
    assert args.embedding_api_key_env == "SILICONFLOW_API_KEY"
    assert args.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert args.embedding_model == "BAAI/bge-m3"
    assert args.embedding_device == "auto"


def test_cli_builds_siliconflow_by_default_without_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_SILICONFLOW_KEY", "test-key")
    received: dict[str, object] = {}

    class _FakeSiliconFlow:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

    monkeypatch.setattr(cli, "SiliconFlowEmbeddingBackend", _FakeSiliconFlow)
    monkeypatch.setattr(
        cli,
        "LocalEmbeddingBackend",
        lambda **kwargs: pytest.fail(f"default must not instantiate local backend: {kwargs}"),
    )

    result = cli.build_embedding_backend(_args())

    assert isinstance(result, _FakeSiliconFlow)
    assert received == {
        "api_key": "test-key",
        "base_url": "https://api.siliconflow.cn/v1",
        "model_name": "BAAI/bge-m3",
        "timeout": 30.0,
    }


def test_cli_local_backend_is_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    class _FakeLocal:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

    monkeypatch.setattr(cli, "LocalEmbeddingBackend", _FakeLocal)
    monkeypatch.setattr(
        cli,
        "SiliconFlowEmbeddingBackend",
        lambda **kwargs: pytest.fail(f"local opt-in must not instantiate API backend: {kwargs}"),
    )

    result = cli.build_embedding_backend(
        _args(embedding_backend="local", local_embedding_model="local/test", embedding_device="cpu")
    )

    assert isinstance(result, _FakeLocal)
    assert received == {"model_name": "local/test", "device": "cpu"}


def test_cli_names_missing_siliconflow_key_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_SILICONFLOW_KEY", raising=False)

    with pytest.raises(SystemExit, match="TEST_SILICONFLOW_KEY"):
        cli.build_embedding_backend(_args())
