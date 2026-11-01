"""Bounded UTF-8 workspace access and reviewable file writes."""
import difflib
import json
import os
from pathlib import Path
from .exclusions import Exclusions, IgnoreConfigurationError

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000


class Workspace:
    def __init__(self, root, *, excludes=()):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")
        self.excludes = tuple(excludes)
        self.exclusions = Exclusions(self.root, self.excludes)

    def path(self, name):
        rel = Path(name)
        if not name or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Use a relative path inside the workspace")
        if any(p in EXCLUDED or p.startswith(".") for p in rel.parts):
            raise ValueError("Hidden and excluded paths are unavailable")
        target = self.root / rel
        current = self.root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlink paths are unavailable")
        target.resolve().relative_to(self.root)
        if self.exclusions.ignored(rel, directory=target.is_dir()):
            raise ValueError(f"Path is excluded by ignore rules: {name}")
        return target

    def files(self, scope=""):
        start = self.path(scope) if scope else self.root
        if not start.is_dir():
            raise ValueError("File listing scope must be a directory")
        found = []
        def visible_directory(path):
            try:
                self.path(path.relative_to(self.root).as_posix())
                return True
            except IgnoreConfigurationError:
                raise
            except (ValueError, OSError):
                return False

        for base, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = sorted(d for d in dirs if visible_directory(Path(base) / d))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                rel = (Path(base) / name).relative_to(self.root).as_posix()
                try:
                    path = self.path(rel)
                    if path.is_file() and path.stat().st_size <= MAX_BYTES:
                        found.append(rel)
                except IgnoreConfigurationError:
                    raise
                except (ValueError, OSError):
                    continue
                if len(found) >= 1000:
                    return found
        return found

    def read(self, name):
        target = self.path(name)
        if not target.is_file():
            raise ValueError("Only regular files can be read")
        with target.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or b"\0" in raw:
            raise ValueError("Only UTF-8 text files up to 64 KB can be read")
        return raw.decode("utf-8")

    def search(self, query, scope=""):
        """Case-sensitive literal search; return bounded, numbered UTF-8 snippets."""
        if (not isinstance(query, str) or not query.strip() or len(query) > 256
                or any(c in query for c in "\n\r\0")):
            raise ValueError("Search requires a nonblank single-line literal of at most 256 characters")
        if scope:
            target = self.path(scope)
            if not target.exists():
                raise ValueError("Search scope does not exist")
            candidates = [scope] if target.is_file() else self.files(scope)
        else:
            candidates = self.files()
        result = {"matches": [], "files_searched": 0, "files_skipped": 0,
                  "limit_reached": False, "candidate_limit_reached": len(candidates) >= 1000}
        for name in candidates:
            try:
                text = self.read(name)
            except (ValueError, OSError, UnicodeError):
                result["files_skipped"] += 1
                continue
            result["files_searched"] += 1
            for number, line in enumerate(text.splitlines(), 1):
                index = line.find(query)
                if index < 0:
                    continue
                start = max(0, index - 40)
                snippet = line[start:start + 320]
                match = {"path": name, "line": number, "text": snippet,
                         "snippet_truncated": start > 0 or start + len(snippet) < len(line)}
                if len(result["matches"]) >= 50:
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
                result["matches"].append(match)
                # Reserve room for counters and the truncation flag on later iterations.
                if len(json.dumps(result, ensure_ascii=False).encode()) > 15_800:
                    result["matches"].pop()
                    result["limit_reached"] = True
                    return json.dumps(result, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def propose(self, name, content):
        if not isinstance(content, str) or len(content.encode()) > MAX_BYTES or "\0" in content:
            raise ValueError("Proposed content must be UTF-8 text up to 64 KB")
        target = self.path(name)
        if target.exists() and target.stat().st_nlink > 1:
            raise ValueError("Hard-linked files cannot be edited")
        before = self.read(name) if target.exists() else None
        diff = "".join(difflib.unified_diff((before or "").splitlines(True),
                       content.splitlines(True), fromfile=name if before is not None else "/dev/null",
                       tofile=name))
        if not diff and before != content:
            diff = f"Create empty file: {name}\n"
        return before, diff

    def write(self, name, content, expected):
        self.propose(name, content)
        target = self.path(name)
        current = self.read(name) if target.exists() else None
        if current != expected:
            raise ValueError("File changed after preview; request a fresh proposal")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
