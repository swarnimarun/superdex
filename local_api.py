"""Loopback Ollama inference API and streaming OpenAI reverse proxy."""
import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from uuid import uuid4

from inkling import MODEL_ID

DETAILS = {"format": "gguf", "family": "inkling2_loop", "families": ["inkling2_loop"],
           "parameter_size": "26.07B"}


def timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def chat_history(messages):
    pending = []
    history = []
    for original in messages:
        msg = original.copy()
        if msg.pop("images", None):
            raise ValueError("This deployment serves text inputs")
        if "thinking" in msg:
            msg["reasoning_content"] = msg.pop("thinking")
        calls = []
        for tool in msg.pop("tool_calls", []):
            ident = tool.get("id") or "call_" + uuid4().hex[:12]
            fn = tool["function"]
            pending.append((fn["name"], ident))
            calls.append({"type": "function", "id": ident, "function": {
                "name": fn["name"], "arguments": json.dumps(fn.get("arguments", {}))}})
        if calls:
            msg["tool_calls"] = calls
        if msg["role"] == "tool":
            name = msg.pop("tool_name", None)
            for item in pending:
                if (not name or item[0] == name) and msg.get("tool_call_id", item[1]) == item[1]:
                    msg["tool_call_id"] = item[1]
                    pending.remove(item)
                    break
        history.append(msg)
    return history


def request_body(body, chat, context):
    options = body.get("options", {})
    req = {"model": MODEL_ID, "stream": True, "stream_options": {"include_usage": True}}
    for name in ("temperature", "top_p", "top_k", "min_p", "seed", "stop", "repeat_penalty"):
        if name in options:
            req[name] = options[name]
    if "num_predict" in options:
        req["max_tokens"] = options["num_predict"]
    if int(options.get("num_ctx", context)) != context:
        raise ValueError("Set INKLING_CONTEXT when starting the server to change context size")
    if body.get("images"):
        raise ValueError("This deployment serves text inputs")
    raw = not chat and body.get("raw", False)
    if raw:
        req["prompt"] = body.get("prompt", "")
    else:
        req["messages"] = chat_history(body["messages"]) if chat else []
        if not chat:
            if body.get("system"):
                req["messages"].append({"role": "system", "content": body["system"]})
            req["messages"].append({"role": "user", "content": body.get("prompt", "")})
        if body.get("tools"):
            req["tools"] = body["tools"]
        if body.get("think") is False:
            req["reasoning_budget_tokens"] = 0
        fmt = body.get("format")
        if fmt:
            req["response_format"] = ({"type": "json_object"} if fmt == "json" else
                                      {"type": "json_schema", "json_schema": {"name": "answer", "schema": fmt}})
    return "/v1/completions" if raw else "/v1/chat/completions", req


def ollama_events(response, chat):
    started = time.monotonic_ns()
    calls, usage, timings = {}, {}, {}
    reason = "stop"

    def event(text="", thinking="", tools=None, done=False):
        item = {"model": MODEL_ID, "created_at": timestamp(), "done": done}
        message = {"role": "assistant", "content": text}
        if thinking:
            message["thinking"] = thinking
        if tools:
            message["tool_calls"] = tools
        if chat:
            item["message"] = message
        else:
            item["response"] = text
            if thinking:
                item["thinking"] = thinking
        return item

    completed = False
    for line in response:
        if line.startswith(b":"):
            # Forward llama-server's SSE heartbeat to idle-sensitive NDJSON clients.
            yield event()
            continue
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            completed = True
            break
        obj = json.loads(data)
        if "error" in obj:
            raise ValueError(str(obj["error"]))
        usage.update(obj.get("usage") or {})
        timings.update(obj.get("timings") or {})
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content") or choice.get("text") or ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if text or thought:
                yield event(text, thought)
            for tool in delta.get("tool_calls", []):
                fn = calls.setdefault(tool["index"], {"name": "", "arguments": ""})
                for key in fn:
                    fn[key] += tool.get("function", {}).get(key) or ""
            reason = choice.get("finish_reason") or reason
    if not completed:
        raise ValueError("Upstream generation ended before its final event")
    if calls:
        yield event(tools=[{"function": {"index": i, "name": fn["name"],
                                         "arguments": json.loads(fn["arguments"] or "{}")}}
                           for i, fn in sorted(calls.items())])
    end = event(done=True)
    end.update(total_duration=time.monotonic_ns() - started, done_reason="length" if reason == "length" else "stop",
               prompt_eval_count=usage.get("prompt_tokens", timings.get("prompt_n", 0)),
               eval_count=usage.get("completion_tokens", timings.get("predicted_n", 0)),
               prompt_eval_duration=int(timings.get("prompt_ms", 0)*1e6),
               eval_duration=int(timings.get("predicted_ms", 0)*1e6))
    yield end


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def proxy(self, method, path, data=None, ollama=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.backend_port, timeout=5)
        started = False
        try:
            conn.connect()
            # A queued request or synchronous cold prefill can take hours.
            conn.sock.settimeout(None if method == "POST" else 30)
            conn.request(method, path, data, {"Content-Type": "application/json"})
            response = conn.getresponse()
            if ollama is None or response.status >= 400:
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.end_headers()
                started = True
                while block := response.read1(16384):
                    self.wfile.write(block)
                    self.wfile.flush()
                return
            chat, stream = ollama
            events = ollama_events(response, chat)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                started = True
                for item in events:
                    self.wfile.write(json.dumps(item, ensure_ascii=False).encode() + b"\n")
                    self.wfile.flush()
            else:
                text, thought, calls = [], [], []
                for item in events:
                    msg = item.get("message", item)
                    text.append(msg.get("content", msg.get("response", "")))
                    thought.append(msg.get("thinking", ""))
                    calls.extend(msg.get("tool_calls", []))
                msg = item["message"] if chat else item
                msg["content" if chat else "response"] = "".join(text)
                if any(thought):
                    msg["thinking"] = "".join(thought)
                if calls:
                    msg["tool_calls"] = calls
                self.send_json(200, item)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if not started:
                self.send_json(502, {"error": str(exc)})
            elif ollama:
                self.wfile.write(json.dumps({"error": str(exc)}).encode() + b"\n")
        finally:
            conn.close()

    def heartbeat(self):
        data = b"Ollama is running"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        if self.path == "/":
            self.heartbeat()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/":
            return self.heartbeat()
        if self.path.startswith("/v1/models?client_version="):
            catalog = json.loads(Path(__file__).with_name("codex-models.json").read_text())
            catalog["models"][0]["context_window"] = self.server.context
            return self.send_json(200, catalog)
        model = self.server.checkpoint
        info = {"name": MODEL_ID, "model": MODEL_ID, "size": model.path.stat().st_size, "digest": model.digest,
                "modified_at": datetime.fromtimestamp(model.path.stat().st_mtime, timezone.utc).isoformat(),
                "details": dict(DETAILS, quantization_level=self.server.checkpoint.quant)}
        if self.path in ("/api/tags", "/api/ps"):
            if self.path == "/api/ps":
                info["context_length"] = self.server.context
                info["superdex"] = self.server.runtime_stats()
            self.send_json(200, {"models": [info]})
        elif self.path == "/api/version":
            self.send_json(200, {"version": "0.1.0", "implementation": "superdex"})
        elif self.path.startswith("/v1/") or self.path == "/health":
            self.proxy("GET", self.path)
        else:
            self.send_json(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 <= size <= 32 * 1024**2:
                return self.send_json(413, {"error": "Request body too large"})
            raw = self.rfile.read(size)
            if self.path in ("/v1/chat/completions", "/v1/chat/completions/input_tokens",
                             "/v1/responses", "/v1/responses/input_tokens", "/v1/completions"):
                payload = json.loads(raw)
                if self.path == "/v1/responses" and payload.get("stream"):
                    # Codex can track typed response.in_progress events through long prefills.
                    payload.setdefault("return_progress", True)
                history = payload.get("input", payload.get("messages", []))
                if isinstance(history, list):
                    for message in history:
                        if isinstance(message, dict) and message.get("role") == "developer":
                            message["role"] = "system"
                return self.proxy("POST", self.path, json.dumps(payload).encode())
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if body.get("model", body.get("name")) not in (MODEL_ID, MODEL_ID.split(":")[0]):
                return self.send_json(404, {"error": "Unknown model"})
            if self.path == "/api/show":
                return self.send_json(200, {"details": dict(DETAILS, quantization_level=self.server.checkpoint.quant), "capabilities": ["completion", "tools", "thinking"],
                    "model_info": {"general.architecture": "inkling2_loop", "inkling2_loop.context_length": self.server.context}})
            if self.path not in ("/api/chat", "/api/generate"):
                return self.send_json(404, {"error": "Unknown endpoint"})
            chat = self.path == "/api/chat"
            path, req = request_body(body, chat, self.server.context)
            self.proxy("POST", path, json.dumps(req).encode(), (chat, body.get("stream", True)))
        except (TypeError, KeyError, ValueError, AttributeError) as exc:
            self.send_json(400, {"error": str(exc)})
