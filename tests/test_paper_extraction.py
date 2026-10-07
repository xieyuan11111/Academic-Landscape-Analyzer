import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.llm_client import OpenAICompatibleClient
from academic_landscape.paper_extraction import PaperExtractionService, PaperExtractionResult


class SDKHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.requests.append(body)
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "paper_id": "paper_from_model",
                                "research_question": {"text": "What is studied?", "evidence_ids": ["E1"]},
                                "claims": [],
                                "evidence": [],
                            }
                        )
                    }
                }
            ]
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):  # noqa: A002
        return


def test_official_openai_sdk_is_used_with_dpsk_controls(tmp_path: Path) -> None:
    SDKHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SDKHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="test-key",
            model="dpsk-test",
        )
        result = client.complete_json("system", "user")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result["paper_id"] == "paper_from_model"
    assert SDKHandler.requests[0]["temperature"] == 0.0
    assert SDKHandler.requests[0]["thinking"] == {"type": "disabled"}
    assert SDKHandler.requests[0]["response_format"] == {"type": "json_object"}


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_paper_extraction_returns_pydantic_result_after_source_validation(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study evaluates method A.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if "evidence anchoring stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {"evidence_id": "E1", "page_number": 1, "hint": "study evaluates method A"}
                    ],
                }
            return {
                "paper_id": document.paper_id,
                "research_question": {"text": "What is evaluated?", "evidence_ids": ["E1"]},
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "Method A is evaluated.",
                        "evidence_type": "B",
                        "evidence_ids": ["E1"],
                    }
                ],
                "concepts": [],
                "theoretical_frameworks": [],
                "methods": [],
                "data_samples": [],
                "contexts": [],
                "results": [],
                "limitations": [],
            }

    client = FakeClient()
    result = PaperExtractionService(client, cache_path=tmp_path / "cache.json").extract(snapshot, document)

    assert isinstance(result, PaperExtractionResult)
    assert result.paper_id == document.paper_id
    assert result.evidence[0].evidence_id == "E1"
    # The anchor locates the sentence; the quote is the whole sentence.
    assert result.evidence[0].quote == "The study evaluates method A."
    assert client.calls == 2  # hint anchoring, then research card


def test_paper_extraction_does_not_fabricate_evidence_from_unlocatable_hint(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "Only a real sentence is present.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def complete_json(self, system_prompt: str, user_prompt: str):
            if "evidence anchoring stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {"evidence_id": "E1", "page_number": 1, "hint": "fabricated phrase absent"}
                    ],
                }
            return {
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

    result = PaperExtractionService(FakeClient(), cache_path=tmp_path / "cache.json").extract(snapshot, document)

    # The hint does not locate a source span, so no fabricated evidence is produced.
    assert result.evidence == []
    assert result.claims == []


def test_paper_extraction_cache_avoids_second_model_call(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study evaluates method A.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, system_prompt: str, user_prompt: str):
            self.calls += 1
            if "evidence anchoring stage" in system_prompt:
                return {"paper_id": document.paper_id, "anchors": []}
            return {
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

    client = FakeClient()
    cache = tmp_path / "cache.json"
    first = PaperExtractionService(client, cache_path=cache).extract(snapshot, document)
    second = PaperExtractionService(client, cache_path=cache).extract(snapshot, document)

    assert first == second
    assert client.calls == 2  # two calls on first extraction; cache serves the second
