"""Doctor self-check command tests.

The promise this module makes to a new user is: "configure it and it works".
These tests lock the three things that promise depends on:

  1. a missing required variable is reported as a blocking failure;
  2. a secret is never printed, only its length;
  3. the exit code is usable by a script (0 ready / 1 not ready).
"""

from __future__ import annotations

import json

import pytest

from academic_landscape.doctor import (
    FAIL,
    OK,
    WARN,
    Check,
    Report,
    main,
    run_doctor,
)

_REQUIRED = (
    "ACADEMIC_LANDSCAPE_BASE_URL",
    "ACADEMIC_LANDSCAPE_MODEL",
    "ACADEMIC_LANDSCAPE_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        *_REQUIRED,
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_EMBEDDING_BASE_URL",
        "SILICONFLOW_EMBEDDING_MODEL",
        "ACADEMIC_LANDSCAPE_NEO4J_URI",
        "ACADEMIC_LANDSCAPE_NEO4J_USER",
        "ACADEMIC_LANDSCAPE_NEO4J_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def full_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_MODEL", "some-model")
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_API_KEY", "sk-SECRET-value-not-to-print")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-ANOTHER-secret-value")


# ----------------------------------------------------------------- verdicts


def test_missing_required_env_fails(clean_env: None) -> None:
    code, report = run_doctor()
    assert code == 1
    failed = {c.name for c in report.failures}
    for name in _REQUIRED:
        assert f"环境变量 {name}" in failed


def test_full_env_is_ready(full_env: None) -> None:
    code, report = run_doctor()
    assert code == 0, report.render()
    assert not report.failures


def test_embedding_key_warns_but_does_not_fail(clean_env: None) -> None:
    """A user who only runs `ingest` offline must not be told the install broke."""
    code, report = run_doctor()
    names = {c.name: c.status for c in report.checks}
    assert names["环境变量 SILICONFLOW_API_KEY"] == WARN
    assert code == 1, "LLM vars are still missing in this scenario"


# ------------------------------------------------------------------- secrecy


def test_secret_value_never_printed(full_env: None, capsys: pytest.CaptureFixture) -> None:
    main([])
    out = capsys.readouterr().out
    assert "sk-SECRET-value-not-to-print" not in out
    assert "sk-ANOTHER-secret-value" not in out


def test_secret_reported_by_length_only(full_env: None, capsys: pytest.CaptureFixture) -> None:
    main([])
    out = capsys.readouterr().out
    assert "字符，不显示内容" in out
    assert "不显示内容" in out


def test_non_secret_values_are_shown(full_env: None, capsys: pytest.CaptureFixture) -> None:
    """A base URL is not a secret; hiding it would only make doctor useless."""
    main([])
    out = capsys.readouterr().out
    assert "https://example.invalid/v1" in out
    assert "some-model" in out


# --------------------------------------------------------------------- output


def test_json_output_is_machine_readable(full_env: None, capsys: pytest.CaptureFixture) -> None:
    code = main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert isinstance(payload["checks"], list)
    assert all({"name", "status", "detail", "hint"} <= set(c) for c in payload["checks"])


def test_json_output_on_failure_has_ok_false(clean_env: None, capsys: pytest.CaptureFixture) -> None:
    code = main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False


def test_every_check_line_has_a_status(capsys: pytest.CaptureFixture) -> None:
    main([])
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.startswith("["):
            assert line[1:5] in {"PASS", "WARN", "FAIL"}, line


# --------------------------------------------------------------------- report


def test_report_render_indents_hints() -> None:
    report = Report()
    report.add("某项", FAIL, "坏了", hint="这样修")
    text = report.render()
    assert "[FAIL] 某项" in text
    assert "-> 这样修" in text


def test_report_render_omits_hint_when_passing() -> None:
    report = Report()
    report.add("某项", OK, "正常", hint="不该出现")
    assert "不该出现" not in report.render()


def test_check_defaults_are_empty() -> None:
    check = Check("x", OK)
    assert check.detail == ""
    assert check.hint == ""


# ----------------------------------------------------------------- cli wiring


def test_cli_exposes_doctor() -> None:
    from academic_landscape.cli import _parser

    args = _parser().parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.probe is False
    assert args.json is False


def test_cli_doctor_forwards_flags() -> None:
    from academic_landscape.cli import _parser

    args = _parser().parse_args(["doctor", "--probe", "--json"])
    assert args.probe is True
    assert args.json is True
