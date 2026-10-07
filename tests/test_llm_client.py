import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from academic_landscape.llm_client import OpenAICompatibleClient


class Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.__class__.requests.append(json.loads(body))
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true, "source": "model"}',
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


def test_openai_compatible_client_posts_json_and_parses_json_response():
    Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="test-key",
            model="test-model",
        )
        result = client.complete_json("system", "user")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result == {"ok": True, "source": "model"}
    assert Handler.requests[0]["model"] == "test-model"
    assert Handler.requests[0]["temperature"] == 0.0
    assert Handler.requests[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in Handler.requests[0]
    assert Handler.requests[0]["response_format"] == {"type": "json_object"}
    assert Handler.requests[0]["messages"][0]["content"] == "system"
