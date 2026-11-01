import argparse
import sys
from .agent import run
from .model import MODEL, Ollama
from .workspace import Workspace


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Offline coding agent powered by local {MODEL}")
    parser.add_argument("task", nargs="?", help="Coding task in plain English")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--apply", action="store_true", help="Allow writing after per-file confirmation")
    parser.add_argument("--max-steps", type=int, default=12, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--doctor", action="store_true", help="Check local server and installed model")
    args = parser.parse_args(argv)
    if not args.doctor and not args.task:
        parser.error("a task is required unless --doctor is used")
    try:
        model = Ollama(endpoint=args.endpoint)
        model.check_model()
        if args.doctor:
            print(f"Local Ollama is ready: {MODEL}")
            return 0
        workspace = Workspace(args.workspace)
        print(f"Workspace: {workspace.root}\nMode: {'confirm each write' if args.apply else 'preview only'}")
        return run(model, workspace, args.task, apply=args.apply, max_steps=args.max_steps,
                   confirm=lambda path: input(f"Write {path}? [y/N] ").strip().lower() == "y")
    except (ValueError, OSError, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
