# Offline Free Coding Agent

Version 0.1.0 — a small command-line coding agent that uses an installed local Ollama model. Ask it to inspect a project, explain code, or propose file changes. No cloud API key, cloud fallback, telemetry, or automatic downloads. The Python application uses only the standard library at runtime.

## Quick start

Requirements: Python 3.10+ and [Ollama](https://ollama.com/). Install Ollama and download a local coding model while connected to the internet. Model quality, speed, memory use, and license depend on your choice; start with a model that fits your device. The example tag below is configurable, not a hardware guarantee.

```sh
python3 scripts/start_ollama.py
# In another terminal, while online:
ollama pull qwen2.5-coder:7b
```

The [startup script](scripts/start_ollama.py) sets `OLLAMA_NO_CLOUD=1` and binds the server to `127.0.0.1:11434`, following [Ollama's local-only instructions](https://docs.ollama.com/faq). Run it from this repository on macOS, Linux, or Windows (use `python` if that is your Python command). Leave that terminal open; Ctrl+C stops the server. Check its logs for `Ollama cloud disabled: true`.

If Ollama is already running, quit its desktop app or stop its service first, then run the script. The script refuses to reuse an occupied port because a running server cannot inherit its environment settings. These settings apply to the server launched by the script; run the script each time you want to start it this way.

After downloading a model, disconnect from the internet to use it offline. Disabling cloud features does not block model downloads or all network traffic. The agent checks that the model is installed and rejects remote-backed model metadata; a loopback URL alone cannot guarantee how a separately configured server performs inference.

From this repository, no Python package installation is required:

```sh
python3 -m offline_agent --model qwen2.5-coder:7b --doctor
python3 -m offline_agent --model qwen2.5-coder:7b \
  --workspace /absolute/path/to/project \
  'Explain this project and suggest a small improvement'
```

By default, changes are previews. To write files, enable `--apply` and type `y` at each file prompt:

```sh
python3 -m offline_agent --model qwen2.5-coder:7b \
  --workspace /absolute/path/to/project --apply \
  'Read the calculator module and add input validation'
```

Optionally install the CLI in a virtual environment (`python3 -m pip install -e .`) and replace `python3 -m offline_agent` with `offline-agent`. Installation may download build tooling; runtime does not require internet access.

## What v0.1 does

- Lists up to 1,000 visible project files and lets the model read selected UTF-8 files.
- Uses structured JSON actions to read, propose a complete file, or finish.
- Shows unified diffs before edits, with confirmation for each write.
- Limits reads/writes to 64 KB per file and tasks to 12 steps by default (maximum 30).
- Rejects traversal, hidden paths, symlinks, and hard-linked write targets.
- Connects only to numeric HTTP loopback endpoints, with proxies and redirects disabled.

Use `--endpoint http://127.0.0.1:11434` for a different local port and `--max-steps 20` for longer tasks. Exit codes: 0 finished/doctor passed, 1 error, 2 step limit, 130 interrupted. Model completion does not establish that generated code is correct.

## Current limits

This is a prototype for trusted local projects, not an operating-system sandbox. It does not execute commands, install dependencies, delete files, run generated tests, or roll back writes. Review changes with Git and run your tests yourself. Earlier confirmed edits remain if a later step fails. Avoid concurrently modifying files or directories while it runs; path checks are not hardened against hostile filesystem races.

Hidden files are skipped, but visible files can still contain secrets. Only point the agent at projects you intend to expose to your local model. It does not yet implement `.gitignore` rules. Conversation context accumulates within a task and can exceed small model context windows; there is no persistent memory or automatic context compaction yet. Diffs for files without trailing newlines can be awkward to read. Model prose is untrusted and may make unsupported claims.

## Development

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q offline_agent tests
```

Tests include a simulated agent workflow and a real loopback HTTP fixture, without downloading a model. A live model smoke test is separate: run `--doctor`, then ask it to create a function in a disposable project, confirm its diff, and test the result.

See [the implementation plan and roadmap](docs/PLAN.md) for architecture, acceptance criteria, and next steps. API integration follows [Ollama chat](https://docs.ollama.com/api/chat) and [structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Initial smoke test: validated locally with `qwen2.5-coder:14b` on Python 3.13.3, including preview and confirmed file creation. All 9 automated tests passed. This does not establish performance on larger projects.
