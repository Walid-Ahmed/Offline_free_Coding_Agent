"""Ollama transport: numeric loopback only, no proxies or redirects."""
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

MODEL = "qwen2.5-coder:7b"


def compact_messages(messages, budget=12_000):
    """Keep a stable prefix until bounded history requires compaction.

    Preserve system instructions and the original task; prefer recent evidence.
    Never mutate the graph's complete in-memory conversation.
    """
    def size(message):
        return len(message["content"].encode("utf-8")) + 64

    if sum(map(size, messages)) <= budget:
        return messages
    if len(messages) < 2:
        raise ValueError("Prompt exceeds budget; shorten the task or increase --prompt-budget")
    fixed = messages[:2]
    remaining = budget - sum(map(size, fixed)) - 160
    if remaining < 512:
        raise ValueError("Task and instructions exceed prompt budget; shorten the task or increase --prompt-budget")
    recent = []
    for message in reversed(messages[2:]):
        if size(message) <= remaining:
            recent.append(message)
            remaining -= size(message)
        elif not recent:
            marker = "\n[CONTEXT TRUNCATED: incomplete evidence; do not reconstruct a file from this excerpt.]"
            available = remaining - len(marker.encode()) - 64
            excerpt = message["content"].encode()[:max(0, available)].decode("utf-8", errors="ignore")
            recent.append({**message, "content": excerpt + marker})
            break
        else:
            break
    notice = {"role": "user", "content": "Earlier context omitted to fit budget; use tools to verify missing evidence."}
    return fixed + [notice] + list(reversed(recent))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model server redirects are disabled")


class Ollama:
    def __init__(self, model=MODEL, endpoint="http://127.0.0.1:11434", timeout=180,
                 max_output_tokens=1024, prompt_budget=12_000):
        if not 256 <= max_output_tokens <= 8192:
            raise ValueError("--max-output-tokens must be between 256 and 8192")
        if not 8000 <= prompt_budget <= 64_000:
            raise ValueError("--prompt-budget must be between 8000 and 64000 bytes")
        self.max_output_tokens = max_output_tokens
        self.prompt_budget = prompt_budget
        url = urllib.parse.urlsplit(endpoint)
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
            url.port  # Validate the optional port before making a request.
        except (ValueError, TypeError):
            local = False
        if (not local or url.scheme != "http" or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP numeric loopback origin, e.g. http://127.0.0.1:11434")
        if model != MODEL:
            raise ValueError(f"Only {MODEL} is supported")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.endpoint + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Model response exceeds 2 MB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("Model server returned a non-object response")
                return result
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(f"Local Ollama request failed: {exc}. Start Ollama and check the installed model.") from exc

    def check_model(self):
        models = self.request("/api/tags").get("models", [])
        names = {m.get("name") for m in models}
        if self.model not in names:
            raise ValueError(f"Model {self.model!r} is not installed. Run: ollama pull {MODEL} while online.")
        info = self.request("/api/show", {"model": self.model})
        if info.get("remote_host") or info.get("remote_model"):
            raise ValueError("Remote-backed models are disabled")

    def chat(self, messages, schema):
        result = self.request("/api/chat", {"model": self.model, "messages": compact_messages(messages, self.prompt_budget),
                              "stream": False, "format": schema,
                              "keep_alive": "30m",
                              "options": {"temperature": 0, "num_predict": self.max_output_tokens,
                                          "num_ctx": max(8192, self.prompt_budget // 2 + self.max_output_tokens)}})
        if result.get("done_reason") == "length":
            raise ValueError("Model reached the response token limit; no partial action was executed. "
                             "Use a smaller task or increase --max-output-tokens.")
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Model returned an invalid JSON response") from exc
