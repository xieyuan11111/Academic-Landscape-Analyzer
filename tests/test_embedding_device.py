"""`device="cuda"` must not be a hard requirement (audit finding D22).

``LocalEmbeddingBackend`` defaulted to ``device="cuda"``, so a CPU-only install
died with ``AssertionError: Torch not compiled with CUDA enabled`` before any
work happened. That is the first-run experience of anyone who installs this
package without a CUDA wheel — and it also bit this project's own 3.13 venv,
where uv installed ``torch==2.14.0+cpu`` while the run script asked for cuda.

Resolution: ``"auto"`` picks cuda when it is genuinely usable and falls back to
cpu otherwise. An explicit ``device="cuda"`` is still honoured (and still fails
loudly) because a caller who names a device means it.
"""

from __future__ import annotations

import sys
import types

import pytest

from academic_landscape.embedding_backend import resolve_device


def _fake_torch(*, available: bool) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: available)
    return module


def test_auto_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True))
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert resolve_device("auto") == "cpu"


def test_auto_falls_back_to_cpu_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing torch must not raise from device resolution itself."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == "cpu"


def test_explicit_device_is_honoured_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming a device is a decision; do not silently override it."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"


def test_default_is_auto_not_cuda() -> None:
    """The constructor default must be portable."""
    import inspect

    from academic_landscape.embedding_backend import LocalEmbeddingBackend

    default = inspect.signature(LocalEmbeddingBackend.__init__).parameters["device"].default
    assert default == "auto", f"default device should be 'auto', got {default!r}"
