"""Offline, in-memory BM25 retrieval over local reference documents."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .workspace import Workspace

EXTENSIONS = {".md", ".txt", ".rst"}


def tokens(text):
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def chunks(text):
    """Keep line citations even when a long line needs multiple chunks."""
    buffer, start, end = [], 1, 1
    size = 0
    for number, line in enumerate(text.splitlines(), 1):
        for offset in range(0, max(1, len(line)), 1600):
            piece = line[offset:offset + 1600]
            if buffer and (size + len(piece) + 1 > 1800 or number - start >= 30):
                yield start, end, "\n".join(buffer)
                buffer, size = [], 0
            if not buffer:
                start = number
            buffer.append(piece)
            size += len(piece) + 1
            end = number
    if buffer:
        yield start, end, "\n".join(buffer)


class KnowledgeBase:
    def __init__(self, directory=None):
        self.documents = 0
        self.skipped = 0
        self.limited = False
        self.entries = []
        self.frequencies = Counter()
        roots = [("builtin", Path(__file__).parent / "references")]
        if directory is not None:
            directory = Path(directory).expanduser()
            if directory.exists():
                roots.append(("user", directory))
            else:
                raise ValueError(f"Knowledge directory does not exist: {directory}")
        total_bytes = 0
        for label, root in roots:
            workspace = Workspace(root)
            names = workspace.files()
            self.limited |= len(names) >= 1000
            for name in names:
                if Path(name).suffix.lower() not in EXTENSIONS:
                    continue
                if self.documents >= 200 or len(self.entries) >= 4000:
                    self.limited = True
                    break
                try:
                    text = workspace.read(name)
                except (ValueError, OSError, UnicodeError):
                    self.skipped += 1
                    continue
                size = len(text.encode())
                if total_bytes + size > 5_000_000:
                    self.limited = True
                    continue
                total_bytes += size
                self.documents += 1
                for start, end, passage in chunks(text):
                    counts = Counter(tokens(passage))
                    if not counts:
                        continue
                    if len(self.entries) >= 4000:
                        self.limited = True
                        break
                    self.entries.append({"source": f"{label}/{name}", "start_line": start,
                                         "end_line": end, "text": passage,
                                         "counts": counts, "length": sum(counts.values())})
                    self.frequencies.update(counts.keys())
        self.average_length = (sum(e["length"] for e in self.entries) / len(self.entries)
                               if self.entries else 1)

    def retrieve(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("Retrieval query must contain 1 to 1000 characters")
        terms = set(tokens(query))
        scored = []
        count = len(self.entries)
        for entry in self.entries:
            score = 0.0
            for term in terms:
                frequency = entry["counts"].get(term, 0)
                if not frequency:
                    continue
                df = self.frequencies[term]
                inverse = math.log(1 + (count - df + 0.5) / (df + 0.5))
                norm = frequency + 1.2 * (0.25 + 0.75 * entry["length"] / self.average_length)
                score += inverse * frequency * 2.2 / norm
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["source"], pair[1]["start_line"]))
        result = {"results": [], "documents": self.documents, "chunks": count,
                  "skipped_documents": self.skipped, "index_limited": self.limited,
                  "results_limited": len(scored) > 4}
        for score, entry in scored[:4]:
            result["results"].append({key: entry[key] for key in
                                      ("source", "start_line", "end_line", "text")})
            result["results"][-1]["score"] = round(score, 4)
            if len(json.dumps(result, ensure_ascii=False).encode()) > 15_800:
                result["results"].pop()
                result["results_limited"] = True
                break
        return json.dumps(result, ensure_ascii=False)
