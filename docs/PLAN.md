# Implementation plan and roadmap

## Product goal

Build a coding assistant that runs with a model on the user's own device, with no paid API or runtime internet dependency. Deliver a small CLI first and expand it after real usage. "Free" describes the absence of a required paid service; hardware, electricity, and model licenses remain the user's responsibility.

## Version 0.1: inspect, propose, review

Original v0.1 architecture (superseded by v0.2 below):

```text
User task + workspace
        |
Python CLI --> bounded agent loop --> local Ollama HTTP API --> installed model
                       |
                validated action
                       |
           read file / preview diff / finish
                       |
             user confirms each write
```

1. Start with Python 3.10+ and no runtime packages for a portable baseline.
2. Use Ollama as the first inference adapter, with the model fixed to `qwen2.5-coder:7b` and no fallback.
3. Enumerate bounded project context; let the model select files to read.
4. Use a JSON action protocol independent of native tool-calling support.
5. Show diffs and require per-file consent in write mode.
6. Validate paths, content size, action shape, step count, and local endpoint.
7. Add offline unit tests, transport integration coverage, and GitHub CI.
8. Publish source and usage guidance in a new repository.

Acceptance criteria: preview mode never writes; declined writes never apply; confirmed writes match the preview; traversal and symlinks are rejected; remote endpoints are rejected; a fake model can complete the read/write/finish workflow; the application reports unavailable Ollama clearly. A live model smoke test depends on an available local service and installed model and must be reported separately from fixture tests.

## Version 0.2: LangGraph orchestration

Implemented a typed in-memory StateGraph with model, read, write, and limit nodes. Conditional edges dispatch validated actions and stop at finish or the model-action limit. The existing workspace restrictions and per-file confirmation remain. LangGraph and LangSmith are installed during setup, with tracing explicitly disabled during runs. There is no hosted graph service, checkpoint persistence, or automatic retry. Inference remains restricted to local `qwen2.5-coder:7b`.

## Next: reliable project editing

Prioritize after trying v0.1 on small real repositories:

- Respect `.gitignore` and user include/exclude rules.
- Add context budgeting and recovery from malformed model output (literal content search shipped in v0.2.2).
- Add multi-file proposals, atomic writes, backups, and rollback.
- Add smaller patch edits with stale-content checks.
- Benchmark `qwen2.5-coder:7b` on representative coding tasks and document tested hardware.

## Version 0.2.1: approved unittest execution

Added a run_tests graph node using fixed unittest discovery under tests/, per-run approval, a 30-second timeout, bounded 16 KB output, and POSIX process-group cleanup. Results feed back into the model conversation. This is explicitly limited to trusted projects on macOS/Linux and is not an isolated runner; test code retains user file/network permissions. Model selection remains fixed to qwen2.5-coder:7b.

## Future: isolated verification loop

Add explicitly approved test commands in an OS-isolated runner, bounded output and timeouts, failure feedback, and clear test evidence. General command execution and confinement of untrusted tests require a separate security design; a fixed command alone is insufficient. Keep network disabled for the runner.

## Later versions

Consider llama.cpp or another local inference adapter, a desktop UI, resumable sessions, repository indexing, and editor integration. Keep the transport boundary and workspace layer separate so these additions do not require rewriting the agent loop.

## Initial validation record

Automated tests exercise simulated model actions and a local HTTP server. See the repository history and CI for results. Actual local inference quality and hardware requirements are not established by these tests.

Local validation on November 1, 2026: all 9 automated tests passed on Python 3.13.3. The installed `qwen2.5-coder:14b` model passed the doctor check, proposed `hello.py` in preview mode, and completed a separate confirmed-write run in a temporary directory. The generated addition function passed two manually executed assertions. This is a small smoke test, not a general coding-quality benchmark. Python 3.10 and Linux checks are configured in CI but were not run locally.

Current model policy: use only `qwen2.5-coder:7b`. The earlier 14b smoke test above is historical and does not validate the current 7b configuration. The CLI no longer accepts a model-selection option.

Version 0.2 validation on November 1, 2026: all 21 tests passed on Python 3.13.3 with LangGraph 1.2.19, and `pip check` reported no broken requirements. A real `qwen2.5-coder:7b` run read a disposable Python file and finished with an explanation in two model actions. Tests also cover the 30-action boundary, approved writes at the limit, and disabled tracing with inherited tracing environment flags. Other Python versions remain covered by the CI configuration rather than local verification.

Version 0.2.1 validation: all 29 automated tests passed on Python 3.13.3. The real local qwen2.5-coder:7b model requested run_tests on a disposable unittest project, received exit code 0 with one passing test, and reported that result. Runner tests cover failures, timeouts, output truncation, missing test directories, rejected custom commands, denied approval, and result feedback into the graph.

## Version 0.2.2: literal file content search

Added a read-only search graph node and scoped Workspace.search. Model arguments reuse content for the query and path for optional file/directory scope. Search is literal and case-sensitive, bounded to 1,000 candidate files, 50 matching lines, and 16 KB of JSON, using existing workspace path/read restrictions. Snippets carry relative paths and 1-based line numbers. No external search program or dependency is needed.

Validation: all 40 automated tests passed on Python 3.13.3, including literal matching, scope, excluded paths, binary/encoding handling, snippets, candidate/result limits, and graph feedback. A live local qwen2.5-coder:7b run called search and correctly reported orders.py at line 1 in a disposable project.
