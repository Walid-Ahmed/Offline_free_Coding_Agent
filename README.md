# Offline Free Coding Agent

![A happy developer coding locally: offline, private, and no API fees.](docs/assets/offline-agent-banner.png)

Version 0.7.0 — a small command-line coding agent that uses only the installed local `qwen2.5-coder:7b` Ollama model. Ask it to inspect a project, explain code, or propose file changes. No cloud API key, cloud fallback, telemetry, or automatic downloads. LangGraph orchestrates the workflow locally; model requests go directly to Ollama. LangSmith tracing is explicitly disabled during agent runs.

## Quick start

Requirements: Python 3.10+ and [Ollama](https://ollama.com/). Install Ollama and download `qwen2.5-coder:7b` while connected to the internet. This is the only supported model; there is no model-selection option or fallback. Speed and memory use depend on your device.

```sh
python3 scripts/start_ollama.py
# In another terminal, while online:
ollama pull qwen2.5-coder:7b
```

The [startup script](scripts/start_ollama.py) sets `OLLAMA_NO_CLOUD=1` and binds the server to `127.0.0.1:11434`, following [Ollama's local-only instructions](https://docs.ollama.com/faq). Run it from this repository on macOS, Linux, or Windows (use `python` if that is your Python command). Leave that terminal open; Ctrl+C stops the server. Check its logs for `Ollama cloud disabled: true`.

If Ollama is already running, quit its desktop app or stop its service first, then run the script. The script refuses to reuse an occupied port because a running server cannot inherit its environment settings. These settings apply to the server launched by the script; run the script each time you want to start it this way.

After downloading a model, disconnect from the internet to use it offline. Disabling cloud features does not block model downloads or all network traffic. The agent checks that the model is installed and rejects remote-backed model metadata; a loopback URL alone cannot guarantee how a separately configured server performs inference.

Install the Python dependencies once while online, then activate the environment each time you open a new terminal:

```sh
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Once installed, the agent runs offline with the downloaded model:

```sh
python3 -m offline_agent --doctor
python3 -m offline_agent \
  --workspace /absolute/path/to/project \
  'Explain this project and suggest a small improvement'
```

By default, changes are previews. To write files, enable `--apply` and type `y` at each file prompt:

```sh
python3 -m offline_agent \
  --workspace /absolute/path/to/project --apply \
  'Read the calculator module and add input validation'
```

After installation, you can replace `python3 -m offline_agent` with `offline-agent` (on Windows, use `python -m offline_agent`). Installation downloads LangGraph and its dependencies; runtime does not require internet access.

## LangGraph workflow

```text
START -> model -> read, search, retrieve, write, or run_tests -> model
             \-> finish -> END
                    tool at action limit -> limit -> END
```

The graph holds conversation messages, the current action, the action count, and the exit code in memory. It uses conditional edges to dispatch actions. Each write still requires your confirmation in `--apply` mode. The action limit counts model responses, not internal graph nodes. Graph recursion has a separate guard sized to that limit. This version does not add persistent checkpoints, saved sessions, or automatic retries of writes.

Implementation follows the [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api). Tracing is disabled using [the tracing context API](https://docs.langchain.com/langsmith/trace-without-env-vars), even if tracing is enabled in the parent environment.

## What v0.7 does

- Lists up to 1,000 visible project files and lets the model read selected UTF-8 files.
- Uses structured JSON actions to read, search file contents, retrieve references, propose a complete file, run tests, or finish.
- Runs unittest or pytest in isolated Docker containers after approval, with networking disabled, read-only project snapshots, a 30-second timeout, and bounded output.
- Shows unified diffs before edits, with confirmation for each write.
- Limits reads/writes to 64 KB per file and tasks to 12 steps by default (maximum 30).
- Rejects traversal, hidden paths, symlinks, and hard-linked write targets.
- Connects only to numeric HTTP loopback endpoints, with proxies and redirects disabled.

Use `--endpoint http://127.0.0.1:11434` for a different local port and `--max-steps 20` for longer tasks. Exit codes: 0 finished/doctor passed, 1 error, 2 step limit, 130 interrupted. Model completion does not establish that generated code is correct.

## File exclusions

File listing, search, RAG indexing, and dependency retrieval now respect root and nested `.gitignore` / `.agentignore` rules. Direct file reads and writes also reject excluded paths. Add persistent agent-only rules to `.agentignore`, or pass extra patterns:

```sh
offline-agent --workspace /path/to/repo \
  --exclude 'private/**' --exclude '*.pem' 'Explain this project'
```

Hidden/built-in paths stay blocked. User exclusions add restrictions; they cannot undo Git ignores. Rules apply to tracked files too. [Exclusion details](docs/EXCLUSIONS.md) cover precedence, negation, root scope, and reference libraries.

## Local reference knowledge base (RAG)

The repository selected by `--workspace` is automatically indexed for RAG. The agent retrieves relevant source code, README/documentation, and common configuration/build files before answering, with citations such as `repo/src/orders.py:1-25`. No separate indexing command is needed. For Python and JavaScript/TypeScript, retrieval also follows local imports up to two hops and adds relevant dependency passages, even when they do not match the question keywords.

```sh
offline-agent --workspace /path/to/repo 'Explain how order validation works'
```

**You can also add your own reference documents.** Put `.md`, `.txt`, or `.rst` files in your project's `knowledge/` folder, or use `--knowledge-base /path/to/references`. A short LangGraph guide is included; you can add locally saved manual pages too. Reference folders inside the workspace must be allowed by its ignore rules. If `knowledge/` is ignored (as in this repository), use an external reference library instead.

```sh
offline-agent --workspace /path/to/project --knowledge-base /path/to/references \
  'Use my references to explain LangGraph conditional edges'
```

The agent retrieves relevant passages automatically and can call `retrieve` for more, with source and line citations. Indexing and retrieval run entirely on your device using BM25 keyword ranking. Startup reports the repository-file count and warns if index limits were reached. The index refreshes after an approved agent write and at the start of each new task. No extra model, internet access, embeddings, or vector database is needed. See [how to add manuals and your own references](docs/KNOWLEDGE_BASE.md) for supported formats, limits, and examples.

## Dependency-aware retrieval and complete functions

If `orders.py` imports a helper from `pricing.py`, retrieval can include the pricing definition too. Python AST and local Tree-sitter JavaScript/TypeScript parsers analyze code without executing it. Supported JS/TS imports include relative ES imports, re-exports, literal `require()`, directory `index` files, and common `.js` specifiers pointing to `.ts` source. Package imports, computed/dynamic paths, and tsconfig/bundler aliases remain unresolved.

Retrieval prefers complete parsed functions, methods, classes, and supported TS declarations when they fit. Results mark `symbol_complete: true` for full symbols or `false` when falling back to a partial excerpt. It considers different files before repeatedly selecting chunks from one file, reserves space for dependencies, and skips candidates that cannot fit instead of ending retrieval immediately.

Up to four keyword matches and four dependency passages share a configurable UTF-8 JSON budget (default 16,000 bytes). For example:

```sh
offline-agent --workspace /path/to/repo --retrieval-budget 8000 \
  'Explain checkout_total and its imported helpers'
```

`--retrieval-budget` accepts 2,000–16,000 bytes. It controls retrieved context only; it does not measure model tokens or cap accumulated conversation history. Imports are followed up to two hops over already indexed files. See [retrieval details](docs/KNOWLEDGE_BASE.md#retrieval-budgets-and-complete-symbols) for metadata and limitations.

## Search file contents

```sh
offline-agent --workspace /absolute/path/to/project 'Search for validate_order and explain where it is used'
```

The model can call `search` with the literal query in `content` and an optional relative file or directory in `path`. An empty path searches the workspace. Search is read-only and needs no approval. Matching is case-sensitive, single-line, and literal (no regex); queries must be nonblank and at most 256 characters.

Results contain relative paths, 1-based line numbers, and up to 320 characters of text around the first match on each matching line. Search returns at most 50 matching lines and 16,000 bytes of JSON. It considers up to 1,000 eligible files per scope and skips hidden/excluded paths, symlinks, files over 64 KB, binary data, unreadable files, and invalid UTF-8. Narrow the scope when a limit is reported. `files_searched` and `files_skipped` describe attempted candidates; `limit_reached` and `candidate_limit_reached` flag potentially incomplete results. A no-match result only describes the files actually searched. Search uses the same ignore and exclusion policy as file listing and RAG.

## Run isolated project tests

Start a local Docker Linux-container runtime and prepare the image once while online:

```sh
docker build -t offline-agent-tests:0.7.0 sandbox
```

Then request either unittest (the default) or pytest:

```sh
offline-agent --workspace /path/to/project --test-runner pytest \
  'Run tests and explain failures'
```

The agent asks for approval before each run. Tests receive a filtered, read-only project copy, disabled networking, and a writable temporary area. They cannot change the original project through the mounted filesystem. Host environment variables and the Docker socket are not exposed. Time/resource/output limits and forced container cleanup apply.

**No host fallback:** Docker and a prepared local image are required. The agent never pulls images or installs packages during a test run. For project dependencies, prepare an image and select `--test-image my-project-tests:local`. See [isolated test setup and limits](docs/ISOLATED_TESTS.md).

## Current limits

This prototype uses Docker isolation for tests; file tools themselves are not an OS sandbox. It runs only the fixed test commands described above; it cannot accept arbitrary commands, install dependencies, delete files through a tool, or roll back writes. Review changes with Git. Earlier confirmed edits remain if a later step fails. Avoid concurrently modifying files or directories while it runs; path checks are not hardened against hostile filesystem races.

Hidden files are skipped, but visible files can still contain secrets. Only point the agent at projects you intend to expose to your local model. Conversation context accumulates within a task and can exceed small model context windows; there is no persistent memory or automatic context compaction yet. Diffs for files without trailing newlines can be awkward to read. Model prose is untrusted and may make unsupported claims.

## Development

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q offline_agent tests
```

Tests include a simulated agent workflow and a real loopback HTTP fixture, without downloading a model. A live model smoke test is separate: run `--doctor`, then ask it to create a function in a disposable project, confirm its diff, and test the result.

See [the implementation plan and roadmap](docs/PLAN.md) for architecture, acceptance criteria, and next steps. API integration follows [Ollama chat](https://docs.ollama.com/api/chat) and [structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Historical v0.1 validation is recorded in [the plan](docs/PLAN.md). Current inference is restricted to `qwen2.5-coder:7b`. Automated tests use fake model responses and a loopback HTTP fixture, so they do not establish coding quality on larger projects.
