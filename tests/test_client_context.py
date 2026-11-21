import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_api import Handler, ollama_events
from scripts import client_config


class ClientContextTests(unittest.TestCase):
    def test_clients_use_live_context_from_a_different_launch_environment(self):
        with tempfile.TemporaryDirectory() as work:
            repo = Path(work)
            shutil.copy(client_config.REPO / "codex-models.json", repo)
            env = {"INKLING_PORT": "11434", "INKLING_CONTEXT": "131072",
                   "INKLING_BASE_URL": "http://127.0.0.1:11434/v1", "INKLING_MODEL": "inkling-2-mini:latest",
                   "PI_CODING_AGENT_DIR": str(repo / "pi")}
            body = {"model_info": {"general.architecture": "inkling2_loop", "inkling2_loop.context_length": 32768}}
            with patch.dict(os.environ, env), patch.object(client_config, "REPO", repo), \
                    patch("http.client.HTTPConnection") as connection:
                for client in ("codex", "pi"):
                    response = io.BytesIO(json.dumps(body).encode())
                    response.status = 200
                    connection.return_value.getresponse.return_value = response
                    self.assertEqual(client_config.configure(client), 32768)
                codex = json.loads((repo / ".runtime/codex-models-32768.json").read_text())
                pi = json.loads((repo / "pi/models.json").read_text())
                self.assertEqual(codex["models"][0]["context_window"], 32768)
                self.assertEqual(pi["providers"]["superdex"]["models"][0]["contextWindow"], 32768)
                connection.return_value.request.side_effect = ConnectionRefusedError()
                self.assertEqual(client_config.context_size(), 131072)

    def test_ollama_stream_heartbeats_keep_the_stream_open(self):
        response = [b":\n", b"\n", b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n',
                    b"data: [DONE]\n"]
        events = list(ollama_events(response, chat=True))
        self.assertEqual([event["done"] for event in events], [False, False, True])
        self.assertEqual([event["message"]["content"] for event in events], ["", "answer", ""])

    def test_responses_progress_and_chat_count_use_the_normalized_prompt(self):
        handler = Mock()
        for path, stream in (("/v1/responses", True), ("/v1/chat/completions/input_tokens", False)):
            history_key = "input" if path == "/v1/responses" else "messages"
            payload = {"stream": stream, history_key: [{"role": "developer", "content": "test"}]}
            data = json.dumps(payload).encode()
            handler.path, handler.headers, handler.rfile = path, {"Content-Length": str(len(data))}, io.BytesIO(data)
            Handler.do_POST(handler)
            method, upstream_path, upstream_bytes = handler.proxy.call_args.args
            upstream = json.loads(upstream_bytes)
            self.assertEqual((method, upstream_path), ("POST", path))
            self.assertEqual(upstream[history_key][0]["role"], "system")
            self.assertEqual(upstream.get("return_progress", False), stream)

    def test_proxy_allows_long_synchronous_generation(self):
        handler = Mock()
        handler.server = SimpleNamespace(backend_port=18567)
        with patch("http.client.HTTPConnection") as connection:
            backend = connection.return_value
            backend.getresponse.return_value.status = 200
            backend.getresponse.return_value.read1.side_effect = [b'{"id":"answer"}', b""]
            Handler.proxy(handler, "POST", "/v1/responses", b'{"stream":false}')
            backend.sock.settimeout.assert_called_once_with(None)
            handler.wfile.write.assert_called_once_with(b'{"id":"answer"}')
            backend.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
