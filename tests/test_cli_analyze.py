import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fitz

from academic_landscape.cli import main
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_cli_analyze_calls_compatible_model_and_writes_report(tmp_path: Path, capsys, monkeypatch) -> None:
    # openai>=2.x refuses to construct a client without credentials even when a
    # base_url is supplied. The CLI resolves the key from this env var, so inject
    # one explicitly instead of relying on the ambient environment.
    monkeypatch.setenv("ACADEMIC_LANDSCAPE_API_KEY", "test-key")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "The paper evaluates method A on outcome B.")
    baseline_dir = tmp_path / "baseline"
    assert main(["ingest", str(input_dir), "--output", str(baseline_dir)]) == 0
    capsys.readouterr()

    snapshot = build_snapshot(input_dir)
    document = snapshot.included_documents[0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            system = request["messages"][0]["content"]
            if "evidence anchoring stage" in system:
                content = {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {"evidence_id": "E1", "page_number": 1, "hint": "paper evaluates method A"}
                    ],
                }
            elif "research card stage" in system:
                content = {
                    "paper_id": document.paper_id,
                    "research_question": {"text": "", "evidence_ids": []},
                    "concepts": [],
                    "theoretical_frameworks": [],
                    "methods": [],
                    "data_samples": [],
                    "contexts": [],
                    "results": [],
                    "limitations": [],
                    "claims": [],
                }
            else:
                content = {
                    "directions": [
                        {
                            "direction_id": "D1",
                            "label": "method evaluation",
                            "paper_ids": [document.paper_id],
                            "evidence_ids": ["E1"],
                            "status": "candidate",
                        }
                    ],
                    "relations": [],
                    "controversies": [],
                    "frontier_candidates": [],
                }
            encoded = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        final_dir = tmp_path / "final"
        assert main(
            [
                "analyze",
                str(baseline_dir),
                "--output",
                str(final_dir),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "test-model",
            ]
        ) == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    result = json.loads(capsys.readouterr().out)
    assert result["semantic_status"] == "validated"
    assert "method evaluation" in (final_dir / "report.md").read_text(encoding="utf-8")
