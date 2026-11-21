"""Small, deterministic checks for local Inkling checkpoint comparisons."""
import json

PROMPTS = [
    "The capital of France is",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return",
    "The next prime after 31 is",
    "Translate 'good morning' into French:",
    "In JSON, an array containing the numbers one, two and three is written",
    "A bicycle has two wheels. Seven bicycles have",
    "The derivative of x squared with respect to x is",
    "To inspect the current Git working tree, run git",
]

def compare(request, reference=None):
    """Use the same reference prefix at each probe, including after a quant diverges."""
    rows = []
    for i, prompt in enumerate(PROMPTS):
        options = {"temperature": 0, "seed": 42, "cache_prompt": False, "ignore_eos": True,
                   "return_tokens": True}
        tokens = request("completion", {"prompt": prompt, "n_predict": 12, **options})["tokens"]
        if len(tokens) != 12:
            raise ValueError("Incomplete sample")
        if reference is None:
            matched = [True] * 3
        else:
            source = reference["samples"][i]
            if source["prompt"] != prompt:
                raise ValueError("Reference prompt differs")
            ids = request("tokenize", {"content": prompt, "add_special": True})["tokens"]
            matched = []
            for pos in (0, 4, 8):
                prefix = ids + source["tokens"][:pos]
                actual = request("completion", {"prompt": prefix, "n_predict": 1, **options})["tokens"]
                matched.append(actual == source["tokens"][pos:pos + 1])
        rows.append({"prompt": prompt, "tokens": tokens, "matched": matched})

    def chat(prompt, **kwargs):
        response = request("v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "seed": 42, "max_tokens": 384, **kwargs})
        return response["choices"][0]["message"]

    checks = {}
    answer = chat("Return only the integer. What is 73 times 19 minus 48?").get("content") or ""
    checks["arithmetic"] = answer.strip() == "1339"
    answer = chat('Return only JSON with key "names" containing Ada and Lin in alphabetical order.').get("content") or ""
    try:
        checks["json"] = json.loads(answer) == {"names": ["Ada", "Lin"]}
    except ValueError:
        checks["json"] = False
    answer = chat("What does Python print? Reply with only the integer.\nprint(sum(x*x for x in range(7) if x % 2))").get("content") or ""
    checks["python"] = answer.strip() == "35"
    fn = {"name": "list_directory", "description": "List the contents of a directory.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
    answer = chat("Use list_directory to list ./src. Pass ./src exactly.", tools=[{"type": "function", "function": fn}])
    calls = answer.get("tool_calls") or []
    try:
        call = calls[0]["function"]
        checks["tool"] = call["name"] == fn["name"] and json.loads(call["arguments"]) == {"path": "./src"}
    except (IndexError, KeyError, TypeError, ValueError):
        checks["tool"] = False
    return {"samples": rows, "checks": checks}
