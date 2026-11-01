"""Ollama transport: numeric loopback only, no proxies or redirects."""
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

MODEL = "qwen2.5-coder:7b"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model server redirects are disabled")


class Ollama:
    def __init__(self, model=MODEL, endpoint="http://127.0.0.1:11434", timeout=180):
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
        result = self.request("/api/chat", {"model": self.model, "messages": messages,
                              "stream": False, "format": schema,
                              "options": {"temperature": 0}})
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Model returned an invalid JSON response") from exc
