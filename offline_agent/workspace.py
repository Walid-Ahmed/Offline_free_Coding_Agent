"""Bounded UTF-8 workspace access and reviewable file writes."""
import difflib
import os
from pathlib import Path

EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
            ".aws", ".ssh", ".codex", ".agents"}
MAX_BYTES = 64_000


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")

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
        return target

    def files(self):
        found = []
        for base, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith(".")
                             and not (Path(base) / d).is_symlink())
            for name in sorted(files):
                if name.startswith("."):
                    continue
                rel = (Path(base) / name).relative_to(self.root).as_posix()
                try:
                    path = self.path(rel)
                    if path.is_file() and path.stat().st_size <= MAX_BYTES:
                        found.append(rel)
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
