# Repository RAG and your own reference documents

The agent has a local reference knowledge base. You can add manuals, team coding
standards, architecture notes, API descriptions, and worked examples. Retrieval
adds relevant passages to the model's context; it does not retrain the model.
No URLs are fetched, no embedding model is downloaded, and indexing never calls
Ollama or any network service. qwen2.5-coder:7b remains the only model.

## Automatic repository RAG

Point `--workspace` at a local repository:

```sh
offline-agent --workspace /path/to/repo 'Explain how authentication works'
```

No copy step or separate index command is required. The selected repository's
supported source code and text documentation are automatically indexed alongside
the bundled guide and your extra references. The initial task retrieves relevant
passages; the model can call `retrieve` for more. Code citations use a `repo/`
prefix, for example `repo/src/auth.py:1-30`. Reference documents under your chosen
knowledge directory keep the `user/` prefix; the same physical file is not indexed
twice. Startup reports repository and reference counts and warns about index limits.

Supported code includes Python, JavaScript/TypeScript, Go, Rust, Java, C/C++, C#,
Ruby, PHP, Swift, Kotlin, shell scripts, SQL, HTML/CSS, Vue/Svelte, and other common
text source formats. Common JSON/YAML/TOML/XML configuration files and build files
such as Dockerfile and Makefile are included. See CODE_EXTENSIONS and CODE_NAMES
in offline_agent/knowledge.py for the exact set. Arbitrary unknown extensions,
images, compiled artifacts, and notebooks are not included.

This is a bounded snapshot, not a guarantee every repository file is indexed.
The existing hidden-path, excluded-directory, UTF-8, file-size, and global index
limits still apply. Indexing does not yet honor `.gitignore`; eligible visible
files may contain private data. Only select repositories you intend to expose to
your local model. Larger repositories may need `search`/`read` for files outside
the index. A directory need not have Git metadata to work.

The index rebuilds after an agent-approved file write, so subsequent retrieval
uses the updated code. Other edits made while a task runs are picked up on the
next task (there is no filesystem watcher). Earlier conversation excerpts remain
snapshots; retrieve or read again when validating changed code.

## Add documents

Create a knowledge folder inside the project you are working on:

```sh
mkdir -p /path/to/project/knowledge
```

Copy your own `.md`, `.txt`, or `.rst` documents into that folder. For example:

```text
my-project/
  knowledge/
    langgraph/
      state-graph.md
      conditional-edges.md
    team-style-guide.md
    architecture.txt
```

Then ask a coding question:

```sh
offline-agent --workspace /path/to/project \
  'Use my references to explain how to route a LangGraph action'
```

Startup tells you where to add references and how many documents were indexed.
The task automatically retrieves relevant passages. The model can request more
using the `retrieve` tool, with its question in `content` and an empty `path`.
Results include source names and line ranges, such as
`user/team-style-guide.md:12-25`. The model is instructed to cite those sources;
review its citations because generation can still make mistakes.

## Use a separate reference library

```sh
offline-agent --workspace /path/to/project \
  --knowledge-base /path/to/my-reference-library \
  'Consult my architecture references before proposing a change'
```

The supplied directory must exist. Documents can be outside the project, but only
the user-configured reference root is indexed. The model cannot select a different
root. Knowledge retrieval reads documents; it never edits that reference library.
If references are inside the workspace, existing file tools can still edit them
with the usual write approval.

## LangGraph manuals

A short, project-authored LangGraph reference ships with the package and is always
included. It is not the full official manual. To add a full manual, place text or
Markdown copies that you already have permission to use in your local knowledge
folder. Split long manuals into smaller files (one topic/page per file), each no
larger than 64 KB. Include the manual version and source URL in the document text
for provenance. URLs remain text; the agent never follows them. There is no
website downloader or automatic documentation updater.

PDF, Word, HTML, and images are not supported by this first version. Export their
useful content to Markdown or UTF-8 plain text before adding it. Only add documents
you intend to expose to your local coding model. In this repository `knowledge/`
is ignored by Git; add the same rule to other projects if references should remain
unpublished.

## How local RAG works

At startup, the agent reads eligible repository and reference files and builds an in-memory BM25
keyword index. It breaks documents into passages of up to 30 lines / about 1,800
characters and ranks them by query term overlap and rarity. It returns up to four
keyword-matched passages plus up to four Python dependency passages with citations
and a shared 16 KB JSON cap. There are no embeddings or semantic
vector search, so queries using the document's actual terminology work best.

The index is rebuilt for each new task and after approved agent writes. External
edits and deletions are picked up on the next run. No database, persistent index, account, or API key is required.
Indexing considers at most 1,000 candidate files per root, 200 documents total across references and repository code,
5 MB of document text, and 4,000 passages. Hidden/excluded paths, symlinks, binary
files, invalid UTF-8, and files over 64 KB are excluded. Metadata flags report
index/result limits; a no-match result is not proof the full library has no answer.
The skipped-document count covers read failures among eligible text candidates,
not every excluded path.

References are untrusted data. The model is instructed to use them as evidence,
not follow embedded commands. This is guidance rather than a prompt-injection
security boundary. Existing write/test approvals still apply. Project test code
retains the previously documented user/network permissions; RAG itself performs
no network requests.

## Dependency-aware Python retrieval

Python repository files (`.py` and `.pyi`) are parsed with Python's standard AST
parser at indexing time. No project module is imported or executed. When a
keyword-matched passage comes from a Python file, retrieval follows its statically
resolved local imports, even if the dependency has no query keyword overlap.

For example, a query matching `orders.py` can return a definition from
`pricing.py` imported with `from pricing import calculate_total`. Absolute imports,
relative imports (`from .pricing ...`, `from ..models ...`), aliases, local package
initializers, and common repository-root / `src/` layouts are supported. A sibling
module is also considered as a fallback for script-style layouts. Resolution only
considers indexed repository paths and never searches installed packages, the
network, or arbitrary files elsewhere on the device.

For a named import, the retriever prefers the passage containing the imported
top-level function, class, or simple assigned constant. If no such definition can
be found, it uses the dependency's first passage. One passage per dependency edge
is selected; importing several symbols does not guarantee that every definition
or entire function body fits. Module imports also default to the first passage.

Expansion follows up to two hops from the keyword-matched files, adding at most
four dependency passages. Cycles and repeated passages are deduplicated. The total
keyword and dependency output shares the 16 KB JSON budget. Dependency results
have `kind: dependency`, `via` (the importing source), and `depth`; normal results
have `kind: match`. `dependency_limit_reached` reports depth/count/output limits;
`dependency_parse_failures` counts Python files that could not be parsed. A syntax
error does not prevent that file's text from participating in keyword retrieval.

This is static import assistance, not a complete Python interpreter or call graph.
Conditional/function-local imports may be included even when not executed. Custom
`sys.path`, dynamic imports, star-import name resolution, runtime monkey-patching,
ambiguous package layouts, reverse caller lookup, and third-party dependencies are
not resolved. Other languages retain keyword-only retrieval. Syntax unsupported by
the Python version running the agent is skipped by dependency analysis. Index
limits may omit import targets; absence of a dependency passage is not evidence
that no dependency exists. Use search/read for missing context.

The dependency graph refreshes with the local index after approved writes and on
each new task. It does not add internet access or another model.
