"""Write a local client catalog using the context of the service being contacted."""
import http.client
import json
import os
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def context_size():
    model = os.environ["INKLING_MODEL"]
    conn = http.client.HTTPConnection("127.0.0.1", int(os.environ["INKLING_PORT"]), timeout=2)
    try:
        conn.request("POST", "/api/show", json.dumps({"model": model}), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status == 200:
            info = json.load(response)["model_info"]
            context = info[f'{info["general.architecture"]}.context_length']
            if type(context) is not int or context <= 0:
                raise ValueError("Invalid context_length in /api/show")
            return context
    except (OSError, http.client.HTTPException):
        pass  # Offline discovery uses the configured startup context.
    finally:
        conn.close()
    return int(os.environ["INKLING_CONTEXT"])


def configure(client):
    context = context_size()
    if client == "pi":
        path = Path(os.environ["PI_CODING_AGENT_DIR"]) / "models.json"
        model = {
            "id": os.environ["INKLING_MODEL"], "name": "Inkling 2 Mini", "reasoning": True,
            "contextWindow": context, "maxTokens": min(8192, context // 2), "input": ["text"],
        }
        config = {"providers": {"superdex": {
            "baseUrl": os.environ["INKLING_BASE_URL"], "api": "openai-completions", "apiKey": "local",
            "compat": {"supportsDeveloperRole": False}, "models": [model],
        }}}
    elif client == "codex":
        config = json.loads((REPO / "codex-models.json").read_text())
        config["models"][0]["context_window"] = context
        path = REPO / ".runtime" / f"codex-models-{context}.json"
    else:
        raise ValueError(f"Unknown client {client}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(config, f)
    os.replace(f.name, path)
    return context


if __name__ == "__main__":
    print(configure(sys.argv[1]))
