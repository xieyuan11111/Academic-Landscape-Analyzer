"""Environment self-check: verify a fresh install can actually run.

Purpose
-------
A new user should not have to discover a missing dependency, an unset key or a
wrong model name by reading a traceback.  ``academic-landscape doctor`` reports
every requirement as a pass / warn / fail line and exits non-zero when a
blocking problem exists, so "configured and ready" is a checkable claim rather
than an assumption.

Design rules
------------
* Never print a secret.  Only whether a variable is *set* and its length.
* Never make a paid API call.  Reachability is reported as a separate, clearly
  labelled ``--probe`` opt-in, because a connectivity check that spends tokens
  silently is worse than no check.
* Read configuration exactly the way the commands do, so a passing doctor
  means the real run will find the same values.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

# Windows consoles default to a legacy code page (cp936/cp1252 depending on the
# machine's locale), so a UTF-8 source file prints as mojibake for anyone whose
# shell does not export a UTF-8 code page.  Reconfigure stdout/stderr when the
# stream is a real console and still on the default encoding.  Guarded so this
# is a no-op under pytest's captured streams and on POSIX.
for _stream in (sys.stdout, sys.stderr):
    _encoding = getattr(_stream, "encoding", None) or ""
    if _stream is not None and hasattr(_stream, "reconfigure") and (
        "utf" not in _encoding.casefold()
    ):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass

OK = "ok"
WARN = "warn"
FAIL = "fail"

_STATUS_MARK = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}

# Packages imported at module top by the core install.  A missing one means
# the install is broken, not merely degraded.
CORE_IMPORTS: tuple[tuple[str, str], ...] = (
    ("fitz", "PyMuPDF —— PDF 解析"),
    ("pydantic", "数据契约校验"),
    ("numpy", "数值计算"),
    ("sklearn", "聚类"),
    ("openai", "LLM / Embedding 客户端"),
    ("qdrant_client", "向量库"),
    ("neo4j", "图数据库驱动"),
)

# Optional extras.  Absent is fine; the report says which feature it gates.
OPTIONAL_IMPORTS: tuple[tuple[str, str, str], ...] = (
    ("torch", "本地离线 embedding", "local-embedding"),
    ("sentence_transformers", "本地离线 embedding", "local-embedding"),
    ("graphiti_core", "Graphiti 时态图投影", "graphiti"),
)

# Environment variables the CLI reads, with whether absence is fatal.
_REQUIRED_ENV: tuple[tuple[str, str], ...] = (
    ("ACADEMIC_LANDSCAPE_BASE_URL", "LLM 端点（OpenAI 兼容）"),
    ("ACADEMIC_LANDSCAPE_MODEL", "LLM 模型名"),
    ("ACADEMIC_LANDSCAPE_API_KEY", "LLM 密钥"),
)
# Needed by `landscape` (the default embedding route), but NOT by `ingest` or
# `apply-analysis`, which are fully offline.  Reported separately so an
# offline-only user is not told their install is broken.
_EMBEDDING_ENV: tuple[tuple[str, str], ...] = (
    ("SILICONFLOW_API_KEY", "默认 embedding 路由密钥；仅 landscape 需要"),
)
_OPTIONAL_ENV: tuple[tuple[str, str], ...] = (
    # NOTE: the base URL and model have real defaults in the code, so they are
    # genuinely optional.  SILICONFLOW_API_KEY has no default and the default
    # embedding route cannot run without it — it is required, and lives in
    # _REQUIRED_ENV, not here.
    ("SILICONFLOW_EMBEDDING_BASE_URL", "embedding 端点（有代码默认值）"),
    ("SILICONFLOW_EMBEDDING_MODEL", "embedding 模型（有代码默认值）"),
)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def render(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = []
        for check in self.checks:
            mark = _STATUS_MARK[check.status]
            line = f"[{mark}] {check.name.ljust(width)}"
            if check.detail:
                line += f"  {check.detail}"
            lines.append(line)
            if check.status != OK and check.hint:
                lines.append(f"       -> {check.hint}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.failures,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "hint": c.hint,
                }
                for c in self.checks
            ],
        }


def _check_python() -> Check:
    version = sys.version_info
    ok = version >= (3, 12)
    return Check(
        name="Python",
        status=OK if ok else FAIL,
        detail=f"{version.major}.{version.minor}.{version.micro}",
        hint="" if ok else "需要 Python 3.12 或更高版本",
    )


def _check_imports(report: Report) -> None:
    for module, purpose in CORE_IMPORTS:
        try:
            importlib.import_module(module)
            report.add(f"依赖 {module}", OK, purpose)
        except Exception as error:  # noqa: BLE001 - report any import failure
            report.add(
                f"依赖 {module}",
                FAIL,
                f"{type(error).__name__}: {error}",
                hint=f"核心依赖缺失，尝试：pip install -e . （{purpose}）",
            )


def _check_optional(report: Report) -> None:
    missing: dict[str, list[str]] = {}
    for module, purpose, extra in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module)
            report.add(f"可选 {module}", OK, purpose)
        except Exception:  # noqa: BLE001 - absence is expected
            report.add(
                f"可选 {module}",
                WARN,
                f"未安装（{purpose}不可用）",
                hint=f"需要该功能时：pip install -e .[{extra}]",
            )
            missing.setdefault(extra, []).append(module)
    for extra, modules in missing.items():
        report.add(
            f"可选组件 [{extra}]",
            WARN,
            f"缺少 {len(modules)} 个包",
            hint=f"pip install -e .[{extra}]",
        )


def _check_env(report: Report) -> None:
    unset_required: list[str] = []
    for name, purpose in _REQUIRED_ENV:
        value = os.environ.get(name, "").strip()
        if not value:
            report.add(
                f"环境变量 {name}",
                FAIL,
                "未设置",
                hint=f"{purpose}。复制 .env.example 为 .env.local 并填入，然后加载到当前 shell",
            )
            unset_required.append(name)
        elif "KEY" in name or "TOKEN" in name or "SECRET" in name:
            # Report presence and length only; never the value.
            report.add(f"环境变量 {name}", OK, f"已设置（{len(value)} 字符，不显示内容）")
        else:
            report.add(f"环境变量 {name}", OK, value)

    for name, purpose in _EMBEDDING_ENV:
        value = os.environ.get(name, "").strip()
        if not value:
            report.add(
                f"环境变量 {name}",
                WARN,
                "未设置（ingest / apply-analysis 不需要，landscape 需要）",
                hint=f"{purpose}。可改用离线路线：--embedding-backend local "
                "（需 pip install -e .[local-embedding]）",
            )
        else:
            report.add(f"环境变量 {name}", OK, f"已设置（{len(value)} 字符，不显示内容）")

    for name, purpose in _OPTIONAL_ENV:
        value = os.environ.get(name, "").strip()
        if not value:
            report.add(
                f"环境变量 {name}",
                WARN,
                "未设置（将使用代码默认值）",
                hint=f"{purpose}；如需自定义请在 .env.local 中设置",
            )
        elif "KEY" in name or "TOKEN" in name or "SECRET" in name:
            report.add(f"环境变量 {name}", OK, f"已设置（{len(value)} 字符，不显示内容）")
        else:
            report.add(f"环境变量 {name}", OK, value)

    if unset_required:
        report.add(
            "下一步",
            FAIL,
            f"{len(unset_required)} 个必需变量未设置",
            hint="cp .env.example .env.local 后填入；Windows PowerShell 用 $env:NAME 导入，"
            "bash 用 export NAME=...",
        )


def _check_cli() -> None:
    """The console script is the documented entry point; confirm it exists."""
    script = shutil.which("academic-landscape")
    Check("命令 academic-landscape", OK if script else WARN, script or "未在 PATH 中")


def _check_neo4j(report: Report, probe: bool) -> None:
    """Neo4j is optional; report reachability without failing the whole run."""
    uri = os.environ.get("ACADEMIC_LANDSCAPE_NEO4J_URI", "").strip()
    if not uri:
        report.add(
            "Neo4j（图投影）",
            WARN,
            "未配置端点，图投影将被跳过",
            hint="可选功能。需要时设置 ACADEMIC_LANDSCAPE_NEO4J_URI，"
            "或改用 --graph-projector 参数",
        )
        return
    if not probe:
        report.add(
            "Neo4j（图投影）",
            OK,
            f"已配置 {uri}（未探测连通性，加 --probe 可测试）",
        )
        return
    try:
        from neo4j import GraphDatabase

        auth = (
            os.environ.get("ACADEMIC_LANDSCAPE_NEO4J_USER", "neo4j"),
            os.environ.get("ACADEMIC_LANDSCAPE_NEO4J_PASSWORD", ""),
        )
        driver = GraphDatabase.driver(uri, auth=auth)
        try:
            driver.verify_connectivity()
            report.add("Neo4j（图投影）", OK, f"{uri} 连通性正常")
        finally:
            driver.close()
    except Exception as error:  # noqa: BLE001 - report any failure
        report.add(
            "Neo4j（图投影）",
            WARN,
            f"{uri} 无法连接：{type(error).__name__}",
            hint="图投影是可选的；分析主体不依赖它",
        )


def run_doctor(probe: bool = False) -> tuple[int, Report]:
    report = Report()
    report.add(
        "自检",
        OK,
        f"academic-landscape-analyzer 环境检查（probe={'on' if probe else 'off'}）",
    )
    report.checks.append(_check_python())
    _check_imports(report)
    _check_optional(report)
    _check_env(report)
    _check_cli()
    _check_neo4j(report, probe)

    if not report.failures:
        report.add("结论", OK, "环境就绪，可以运行 ingest / analyze / landscape")
    else:
        report.add(
            "结论",
            FAIL,
            f"{len(report.failures)} 项检查未通过",
            hint="按上面的提示补齐后重新运行：academic-landscape doctor",
        )
    return (0 if not report.failures else 1), report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="academic-landscape doctor",
        description="检查环境、依赖与配置是否就绪（不会打印密钥，不消耗 API 额度）",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="额外测试 Neo4j 连通性（不消耗 LLM 额度）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本判断")
    args = parser.parse_args(argv)

    code, report = run_doctor(probe=args.probe)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
        print()
        print(f"退出码 {code}（0=就绪，1=有必须修复的问题）")
    return code
