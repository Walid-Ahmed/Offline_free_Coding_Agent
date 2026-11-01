import argparse
from functools import partial
import sys
from pathlib import Path
from .knowledge import KnowledgeBase
from .agent import run
from .model import MODEL, Ollama
from .workspace import Workspace
from .test_runner import DEFAULT_IMAGE


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"CodeHaven — Your code. Your machine. No API fees. Powered by local {MODEL}")
    parser.add_argument("task", nargs="?", help="Coding task in plain English")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Additional Git-style exclusion; repeat for multiple patterns")
    parser.add_argument("--knowledge-base", help="Local reference directory; defaults to workspace/knowledge")
    parser.add_argument("--retrieval-budget", type=int, default=16000, metavar="BYTES",
                        help="Retrieval JSON budget in UTF-8 bytes (2000..16000; default 16000)")
    parser.add_argument("--test-runner", choices=("unittest", "pytest"), default="unittest")
    parser.add_argument("--test-image", default=DEFAULT_IMAGE, help="Prepared local Docker image for isolated tests")
    parser.add_argument("--test-docker-socket", help="Absolute local Unix Docker socket; remote daemons are unsupported")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--apply", action="store_true", help="Allow writing after per-file confirmation")
    parser.add_argument("--max-steps", type=int, default=12, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--doctor", action="store_true", help="Check local server and installed model")
    args = parser.parse_args(argv)
    if not 2000 <= args.retrieval_budget <= 16000:
        parser.error("--retrieval-budget must be between 2000 and 16000 bytes")
    if not args.doctor and not args.task:
        parser.error("a task is required unless --doctor is used")
    try:
        model = Ollama(endpoint=args.endpoint)
        model.check_model()
        if args.doctor:
            print(f"Local Ollama is ready: {MODEL}")
            return 0
        workspace = Workspace(args.workspace, excludes=args.exclude)
        references = Path(args.knowledge_base).expanduser() if args.knowledge_base else workspace.root / "knowledge"
        knowledge = KnowledgeBase(references if args.knowledge_base or references.exists() else None,
                                  workspace=workspace, context_budget=args.retrieval_budget)
        print(f"Knowledge base: {knowledge.documents_by_source['repo']} repository files, "
              f"{knowledge.documents_by_source['user']} user references, {len(knowledge.entries)} chunks. "
              "Add your own .md, .txt, or .rst references to " + str(references))
        if knowledge.limited:
            print("Index limits reached; retrieval covers only part of the available files. Use search/read for more.")
        if knowledge.excluded_reference_roots:
            print("Reference directory excluded by workspace rules; it was not indexed. "
                  "Use a permitted directory or an external reference library.")
        print(f"Workspace: {workspace.root}\nFile edits: {'confirm each write' if args.apply else 'preview only'}")
        return run(model, workspace, args.task, apply=args.apply, max_steps=args.max_steps, knowledge=knowledge,
                   output=partial(print, flush=True),
                   test_options={"runner": args.test_runner, "image": args.test_image,
                                 "docker_socket": args.test_docker_socket},
                   confirm=lambda path: input(f"Write {path}? [y/N] ").strip().lower() == "y",
                   confirm_tests=lambda: input(
                       f"Run {args.test_runner} in Docker with network disabled and project read-only? [y/N] "
                   ).strip().lower() == "y")
    except (ValueError, OSError, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
