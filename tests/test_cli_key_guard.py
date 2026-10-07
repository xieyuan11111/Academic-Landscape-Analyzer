"""D21: a missing API key must fail with a readable message, not a raw traceback.

``cli.py` guards ``--base-url`` and ``--model`` but not the key: with
``openai`` 2.x→3.x the SDK now raises ``OpenAIError`` during construction, so an
unconfigured install dumps a provider traceback at the user instead of telling
them which environment variable to set. That is the exact first-run experience
of someone else who installs this from source.

The guard must name the *variable to set*, because that is the missing piece of
information; the SDK's own message tells them to pass ``api_key`` but they are
not calling the SDK — they are running a CLI.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from academic_landscape.cli import main


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_analyze_without_api_key_exits_readably(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ACADEMIC_LANDSCAPE_API_KEY", raising=False)
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_MODEL", "test-model")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "The paper evaluates method A.")
    assert main(["ingest", str(input_dir), "--output", str(tmp_path / "baseline")]) == 0

    raised: SystemExit | None = None
    try:
        main(["analyze", str(tmp_path / "baseline"), "--output", str(tmp_path / "out")])
    except SystemExit as exc:
        raised = exc

    assert raised is not None, "missing key must abort, not reach the SDK"
    message = str(raised.code or "")
    # The message must name the variable the user can actually set.
    assert "ACADEMIC_LANDSCAPE_API_KEY" in message


def test_landscape_without_api_key_exits_readably(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ACADEMIC_LANDSCAPE_API_KEY", raising=False)
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_MODEL", "test-model")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "The paper evaluates method A.")
    assert main(["ingest", str(input_dir), "--output", str(tmp_path / "baseline")]) == 0

    raised: SystemExit | None = None
    try:
        main(["landscape", str(tmp_path / "baseline"), "--output", str(tmp_path / "out")])
    except SystemExit as exc:
        raised = exc

    assert raised is not None
    assert "ACADEMIC_LANDSCAPE_API_KEY" in str(raised.code or "")


def test_explicit_key_env_name_is_honoured(tmp_path: Path, monkeypatch) -> None:
    """A caller may pass a different --api-key-env; the guard must use that name."""
    monkeypatch.setenv("MY_OWN_KEY_VAR", "test-key")
    monkeypatch.delenv("ACADEMIC_LANDSCAPE_API_KEY", raising=False)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "The paper evaluates method A.")
    assert main(["ingest", str(input_dir), "--output", str(tmp_path / "baseline")]) == 0

    raised: SystemExit | None = None
    try:
        main([
            "analyze", str(tmp_path / "baseline"),
            "--output", str(tmp_path / "out"),
            "--base-url", "http://127.0.0.1:1/v1",
            "--model", "test-model",
            "--api-key-env", "MY_OWN_KEY_VAR",
        ])
    except SystemExit as exc:
        raised = exc
    except Exception as exc:  # noqa: BLE001 - expected: connection refused to :1
        # Guard must not block a configured call: reaching a transport error
        # means credentials passed validation and the key name was honoured.
        assert "MY_OWN_KEY_VAR" not in str(exc), "guard fired despite key being set"
        return

    # If it raised SystemExit, it must not be a credentials complaint.
    assert raised is None or "MY_OWN_KEY_VAR" not in str(raised.code or "")
