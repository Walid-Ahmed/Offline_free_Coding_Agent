"""Bounded UTF-8 workspace access and reviewable file writes."""
import ast
import difflib
import json
import os
import re
from pathlib import Path
from .exclusions import Exclusions, IgnoreConfigurationError

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000
SEARCH, DIVIDER, REPLACE = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"


def line_range(spec):
    """Parse an optional read range such as 120-180 or 120; anything else means the whole file."""
    match = re.fullmatch(r"\s*(?:lines?\s*)?(\d+)(?:\s*[-:]\s*(\d+))?\s*", spec or "", re.I)
    if not match:
        return None
    start, end = int(match[1]), int(match[2]) if match[2] else None
    if start < 1 or (end is not None and end < start):
        raise ValueError("Read range must be start-end with 1 <= start <= end")
    return start, end


def python_syntax_error(name, content, before=None):
    """Describe a syntax error a change introduces into a Python file. Parses only; never executes."""
    if Path(name).suffix not in (".py", ".pyi"):
        return None
    try:
        ast.parse(content, filename=name)
        return None
    except SyntaxError as exc:
        error = exc
    except (ValueError, RecursionError, MemoryError):
        return None
    if before is not None:
        try:
            ast.parse(before, filename=name)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None  # Already broken before this change; do not block a repair attempt.
    line = (error.text or "").strip()
    return (f"Proposed {name} has a Python syntax error at line {error.lineno}: {error.msg}"
            + (f" ({line[:120]})" if line else "") + ". Nothing was previewed or written; fix the change and try again.")


def apply_edits(text, blocks):
    """Apply SEARCH/REPLACE blocks in order; each SEARCH text must match exactly once."""
    lines = blocks.split("\n")
    edits, index = [], 0

    def marker(name, start):
        found = next((i for i in range(start, len(lines)) if lines[i].strip() == name), None)
        if found is None:
            raise ValueError(f"Each edit block needs {DIVIDER} and {REPLACE} marker lines")
        return found

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index].strip() != SEARCH:
            raise ValueError(f"Edit content must contain only {SEARCH} ... {REPLACE} blocks")
        divider = marker(DIVIDER, index + 1)
        end = marker(REPLACE, divider + 1)
        edits.append(("\n".join(lines[index + 1:divider]), "\n".join(lines[divider + 1:end])))
        index = end + 1
    if not edits:
        raise ValueError("Edit needs at least one SEARCH/REPLACE block")
    newline = "\r\n" if "\r\n" in text else "\n"
    for number, (old, new) in enumerate(edits, 1):
        if not old.strip():
            raise ValueError(f"Edit block {number} has empty SEARCH text; use write to create a file")
        old, new = old.replace("\n", newline), new.replace("\n", newline)
        count = text.count(old)
        if count != 1:
            hint = ("read the file and copy the current lines exactly" if not count
                    else "include more surrounding lines so it matches once")
            raise ValueError(f"Edit block {number} SEARCH text matches {count} times; {hint}. No change was made.")
        text = text.replace(old, new, 1)
    return text


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

    def page(self, name, spec="", limit=MAX_BYTES):
        """Return (text, start, end, total) for a line range, cut at a line boundary to fit limit bytes."""
        text = self.read(name)
        lines = text.splitlines(keepends=True)
        total = len(lines)
        requested = line_range(spec)
        if requested is None and len(text.encode()) <= limit:
            return text, 1, max(1, total), total
        start, end = requested or (1, None)
        if start > total:
            raise ValueError(f"{name} has {total} lines; request a range within 1-{total}")
        shown, size = [], 0
        for line in lines[start - 1:min(end or total, total)]:
            size += len(line.encode())
            if shown and size > limit:
                break
            shown.append(line)
        return "".join(shown), start, start + len(shown) - 1, total

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
        result = {"query": query, "scope": scope or ".", "literal": True,
                  "matches": [], "files_searched": 0, "files_skipped": 0,
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
        if not result["matches"]:
            result["note"] = ("No matches for this exact literal and scope among searched files. "
                              "This does not establish that a framework is absent. "
                              "Do not append file extensions or globs to a source-text query.")
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
