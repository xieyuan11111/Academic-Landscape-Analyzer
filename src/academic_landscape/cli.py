"""Command-line entry point for the academic-landscape analyzer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .analysis import apply_analysis
from .embedding_backend import LocalEmbeddingBackend, SiliconFlowEmbeddingBackend
from .engine import AnalysisEngine
from .ingest import build_snapshot
from .landscape_report import (
    landscape_result_to_dict,
    render_landscape_report,
)
from .llm_client import OpenAICompatibleClient
from .paper_extraction import PaperExtractionService
from .pipeline import LandscapePipeline
from .report import write_report
from .snapshot import load_snapshot


def _require_api_key(api_key_env: str) -> None:
    """Abort with the variable to set when credentials are missing.

    The openai SDK raises ``OpenAIError`` during client construction with no
    ``api_key``, which leaves the caller staring at a provider traceback instead
    of the one fact that unblocks them. The CLI resolves the key from an
    environment variable, so name that variable in the failure.
    """
    if not os.environ.get(api_key_env, "").strip():
        raise SystemExit(
            f"missing API key: set the {api_key_env} environment variable "
            "(or pass --api-key-env with the name of one that is set)"
        )


def build_embedding_backend(args: argparse.Namespace) -> object:
    """Construct the explicitly selected embedding route for ``landscape``.

    SiliconFlow is intentionally the default.  A provider error is surfaced to
    the caller; there is no automatic local fallback because mixing models would
    silently change Qdrant dimensions and make runs incomparable.
    """
    if args.embedding_backend == "siliconflow":
        _require_api_key(args.embedding_api_key_env)
        return SiliconFlowEmbeddingBackend(
            api_key=os.environ.get(args.embedding_api_key_env, ""),
            base_url=args.embedding_base_url,
            model_name=args.embedding_model,
            timeout=args.embedding_timeout,
        )
    if args.embedding_backend == "local":
        return LocalEmbeddingBackend(
            model_name=args.local_embedding_model,
            device=args.embedding_device,
        )
    raise AssertionError(f"Unhandled embedding backend: {args.embedding_backend}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="academic-landscape")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="build a corpus snapshot from PDF files")
    ingest.add_argument("input_dir", type=Path)
    ingest.add_argument("--output", "-o", type=Path, required=True)

    apply = subparsers.add_parser(
        "apply-analysis",
        help="validate an AI analysis JSON against a snapshot and render a report",
    )
    apply.add_argument("snapshot_dir", type=Path)
    apply.add_argument("analysis_json", type=Path)
    apply.add_argument("--output", "-o", type=Path, required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="call an OpenAI-compatible model, validate its output and render a report",
    )
    analyze.add_argument("snapshot_dir", type=Path)
    analyze.add_argument("--output", "-o", type=Path, required=True)
    analyze.add_argument("--base-url", default=os.environ.get("ACADEMIC_LANDSCAPE_BASE_URL"))
    analyze.add_argument("--model", default=os.environ.get("ACADEMIC_LANDSCAPE_MODEL"))
    analyze.add_argument(
        "--api-key-env",
        default="ACADEMIC_LANDSCAPE_API_KEY",
        help="environment variable containing the provider key",
    )
    analyze.add_argument("--timeout", type=float, default=120.0)
    analyze.add_argument(
        "--extraction-cache",
        type=Path,
        help="durable cache for validated single-paper extraction results",
    )

    landscape = subparsers.add_parser(
        "landscape",
        help="run the knowledge-base-driven landscape pipeline and render the §9 report",
    )
    landscape.add_argument("snapshot_dir", type=Path)
    landscape.add_argument("--output", "-o", type=Path, required=True)
    landscape.add_argument("--base-url",
                           default=os.environ.get("ACADEMIC_LANDSCAPE_BASE_URL"))
    landscape.add_argument("--model",
                           default=os.environ.get("ACADEMIC_LANDSCAPE_MODEL"))
    landscape.add_argument(
        "--api-key-env",
        default="ACADEMIC_LANDSCAPE_API_KEY",
        help="environment variable containing the provider key",
    )
    landscape.add_argument("--timeout", type=float, default=120.0)
    landscape.add_argument(
        "--extraction-cache",
        type=Path,
        help="durable cache for validated single-paper extraction results",
    )
    landscape.add_argument(
        "--embedding-backend",
        choices=("siliconflow", "local"),
        default="siliconflow",
        help="embedding route: SiliconFlow API by default; local is an explicit offline opt-in",
    )
    landscape.add_argument(
        "--embedding-api-key-env",
        default="SILICONFLOW_API_KEY",
        help="environment variable containing the embedding-provider API key",
    )
    landscape.add_argument(
        "--embedding-base-url",
        default=os.environ.get(
            "SILICONFLOW_EMBEDDING_BASE_URL", SiliconFlowEmbeddingBackend.DEFAULT_BASE_URL
        ),
        help="OpenAI-compatible embedding endpoint for --embedding-backend siliconflow",
    )
    landscape.add_argument(
        "--embedding-model",
        default=os.environ.get("SILICONFLOW_EMBEDDING_MODEL", SiliconFlowEmbeddingBackend.DEFAULT_MODEL),
        help="embedding model for --embedding-backend siliconflow",
    )
    landscape.add_argument("--embedding-timeout", type=float, default=30.0)
    landscape.add_argument(
        "--local-embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformer model used only with --embedding-backend local",
    )
    landscape.add_argument("--embedding-device", default="auto")
    landscape.add_argument("--current-year", type=int, default=2026)
    landscape.add_argument("--n-clusters", type=int, default=None)

    doctor = subparsers.add_parser(
        "doctor",
        help="check that the environment, dependencies and configuration are ready",
    )
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="also test Neo4j connectivity (never spends LLM quota)",
    )
    doctor.add_argument(
        "--json", action="store_true", help="machine-readable output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        from .doctor import main as doctor_main

        forwarded: list[str] = []
        if args.probe:
            forwarded.append("--probe")
        if args.json:
            forwarded.append("--json")
        return doctor_main(forwarded)
    if args.command == "ingest":
        snapshot = build_snapshot(args.input_dir)
        paths = write_report(snapshot, args.output)
        print(
            json.dumps(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "document_count": len(snapshot.documents),
                    "included_document_count": len(snapshot.included_documents),
                    "output_dir": str(paths.output_dir),
                    "json": str(paths.json_path),
                    "markdown": str(paths.markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "apply-analysis":
        paths, validation = apply_analysis(args.snapshot_dir, args.analysis_json, args.output)
        print(
            json.dumps(
                {
                    "snapshot_id": validation.payload["snapshot_id"],
                    "semantic_status": "validated",
                    "output_dir": str(paths.output_dir),
                    "json": str(paths.json_path),
                    "markdown": str(paths.markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "analyze":
        if not args.base_url:
            raise SystemExit("analyze requires --base-url or ACADEMIC_LANDSCAPE_BASE_URL")
        if not args.model:
            raise SystemExit("analyze requires --model or ACADEMIC_LANDSCAPE_MODEL")
        _require_api_key(args.api_key_env)
        snapshot = load_snapshot(args.snapshot_dir)
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env, ""),
            model=args.model,
            timeout=args.timeout,
        )
        extraction_cache = args.extraction_cache or (args.snapshot_dir / "paper_extraction_cache.json")
        analysis = AnalysisEngine(client, cache_path=extraction_cache).run(snapshot)
        paths = write_report(snapshot, args.output, analysis=analysis)
        print(
            json.dumps(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "semantic_status": "validated",
                    "output_dir": str(paths.output_dir),
                    "json": str(paths.json_path),
                    "markdown": str(paths.markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "landscape":
        if not args.base_url:
            raise SystemExit("landscape requires --base-url or ACADEMIC_LANDSCAPE_BASE_URL")
        if not args.model:
            raise SystemExit("landscape requires --model or ACADEMIC_LANDSCAPE_MODEL")
        _require_api_key(args.api_key_env)
        snapshot = load_snapshot(args.snapshot_dir)
        model_client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env, ""),
            model=args.model,
            timeout=args.timeout,
        )
        extraction_cache = args.extraction_cache or (
            args.snapshot_dir / "paper_extraction_cache.json"
        )
        extraction_service = PaperExtractionService(
            model_client, cache_path=extraction_cache
        )
        extractions = {}
        for document in snapshot.included_documents:
            result = extraction_service.extract(snapshot, document)
            extractions[document.paper_id] = result.model_dump(mode="json")

        embedding_backend = build_embedding_backend(args)
        pipeline = LandscapePipeline(
            client=model_client,
            embedding_backend=embedding_backend,
            current_year=args.current_year,
            n_clusters=args.n_clusters,
        )
        landscape = pipeline.run(snapshot, extractions)

        output_dir = args.output.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "landscape.json"
        markdown_path = output_dir / "landscape.md"
        json_path.write_text(
            json.dumps(landscape_result_to_dict(landscape), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(
            render_landscape_report(landscape), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "directions": len(landscape.directions),
                    "frontier_signals": len(landscape.merged_frontiers),
                    "output_dir": str(output_dir),
                    "json": str(json_path),
                    "markdown": str(markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
