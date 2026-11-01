"""Local evaluation harness: run fixed tasks on a sample repository and score the local model.

Usage: python -m offline_agent.evaluation [--only ID] [--output PATH]

Each task runs on a fresh temporary copy of the sample repository, with writes auto-confirmed
inside that copy and test runs declined. The model is reached only through loopback Ollama,
so evaluation is as offline as normal use. Results record pass/fail, exit code, steps, tokens,
and time, so prompt or tool changes can be compared run to run.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_state
from .knowledge import KnowledgeBase
from .model import MODEL, Ollama
from .workspace import Workspace

EVALS = Path(__file__).resolve().parents[1] / "evals"


class Metered:
    """Wrap a model and add up the calls, tokens, and model time Ollama reports."""

    def __init__(self, model):
        self.model = model
        self.prompt_budget = getattr(model, "prompt_budget", 12_000)
        self.last_metrics = {}
        self.calls = self.prompt_tokens = self.generated_tokens = 0
        self.model_seconds = 0.0

    def chat(self, messages, schema):
        self.calls += 1
        try:
            return self.model.chat(messages, schema)
        finally:
            metrics = getattr(self.model, "last_metrics", None)
            self.last_metrics = metrics if isinstance(metrics, dict) else {}
            self.prompt_tokens += self.last_metrics.get("prompt_tokens") or 0
            self.generated_tokens += self.last_metrics.get("generated_tokens") or 0
            self.model_seconds += self.last_metrics.get("total_seconds") or 0


def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must hold a non-empty JSON list")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not isinstance(task.get("task"), str):
            raise ValueError(f"Each task needs string id and task fields: {task!r:.120}")
        if task["id"] in seen:
            raise ValueError(f"Duplicate task id: {task['id']}")
        seen.add(task["id"])
        checks = task.get("checks")
        if not isinstance(checks, dict) or not (checks.get("answer") or checks.get("files")):
            raise ValueError(f"Task {task['id']} needs answer or files checks")
        for pattern in checks.get("answer", []) + [p for ps in checks.get("files", {}).values() for p in ps]:
            re.compile(pattern)
    return tasks


def score(task, answer, root):
    """Return failed checks: answer regexes (case-insensitive) and file regexes after the run."""
    failures = [f"answer lacks /{pattern}/" for pattern in task["checks"].get("answer", [])
                if not re.search(pattern, answer, re.I | re.S)]
    for name, patterns in task["checks"].get("files", {}).items():
        path = Path(root, name)
        if not path.is_file():
            failures.append(f"{name} missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures += [f"{name} lacks /{pattern}/" for pattern in patterns if not re.search(pattern, text, re.M)]
    return failures


def evaluate_task(task, model, repo):
    with tempfile.TemporaryDirectory(prefix="codehaven-eval-") as temporary:
        root = Path(temporary) / "repo"
        shutil.copytree(repo, root)
        workspace = Workspace(root)
        metered, log = Metered(model), []
        started = time.monotonic()
        answer, error = "", None
        try:
            state = run_state(metered, workspace, task["task"], apply=task.get("apply", False),
                              max_steps=task.get("max_steps", 12),
                              knowledge=KnowledgeBase(None, workspace=workspace),
                              confirm=lambda _path: True, confirm_tests=lambda: False, output=log.append)
            exit_code, steps = state["exit_code"], state["steps"]
            reply = state.get("reply") or {}
            if reply.get("action") == "finish":
                answer = reply.get("content") or reply.get("message") or ""
        except ValueError as exc:
            exit_code, steps, error = 1, metered.calls, str(exc)
        failures = ([f"error: {error}"] if error else []) + score(task, answer, root)
    return {"id": task["id"], "passed": not failures, "exit_code": exit_code, "steps": steps,
            "prompt_tokens": metered.prompt_tokens, "generated_tokens": metered.generated_tokens,
            "model_seconds": round(metered.model_seconds, 2), "seconds": round(time.monotonic() - started, 2),
            "failures": failures, "answer": answer[:2000]}


def summarize(results):
    count = len(results)
    return {"tasks": count, "passed": sum(r["passed"] for r in results),
            "mean_steps": round(sum(r["steps"] for r in results) / count, 2) if count else 0,
            "prompt_tokens": sum(r["prompt_tokens"] for r in results),
            "generated_tokens": sum(r["generated_tokens"] for r in results),
            "seconds": round(sum(r["seconds"] for r in results), 1)}


def table(results):
    width = max([len("TASK")] + [len(r["id"]) for r in results])
    lines = [f"{'TASK':<{width}}  PASS  EXIT  STEPS  PROMPT TOK  GEN TOK  SECONDS  NOTES"]
    for r in results:
        lines.append(f"{r['id']:<{width}}  {'yes' if r['passed'] else 'NO':<4}  {r['exit_code']:<4}  {r['steps']:<5}  "
                     f"{r['prompt_tokens']:>10,}  {r['generated_tokens']:>7,}  {r['seconds']:>7.1f}  "
                     + "; ".join(r["failures"])[:120])
    return "\n".join(lines)


def evaluate(tasks, model, repo, output=print):
    results = []
    for number, task in enumerate(tasks, 1):
        output(f"[{number}/{len(tasks)}] {task['id']}: {task['task']}")
        results.append(evaluate_task(task, model, repo))
        result = results[-1]
        output(f"    {'PASS' if result['passed'] else 'FAIL'} in {result['steps']} steps, {result['seconds']:.1f}s"
               + (f" ({'; '.join(result['failures'])})" if result["failures"] else ""))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Score local {MODEL} on fixed CodeHaven tasks (offline)")
    parser.add_argument("--tasks", default=str(EVALS / "tasks.json"))
    parser.add_argument("--repo", default=str(EVALS / "sample_repo"))
    parser.add_argument("--only", action="append", default=[], metavar="ID", help="Run only this task id; repeatable")
    parser.add_argument("--output", default=str(EVALS / "results" / "latest.json"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--prompt-budget", type=int, default=12000)
    args = parser.parse_args(argv)
    try:
        tasks = load_tasks(args.tasks)
        if args.only:
            unknown = set(args.only) - {task["id"] for task in tasks}
            if unknown:
                parser.error("unknown task id: " + ", ".join(sorted(unknown)))
            tasks = [task for task in tasks if task["id"] in args.only]
        if not Path(args.repo).is_dir():
            raise ValueError(f"Sample repository not found: {args.repo}")
        model = Ollama(endpoint=args.endpoint, max_output_tokens=args.max_output_tokens, prompt_budget=args.prompt_budget)
        model.check_model()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results = evaluate(tasks, model, args.repo, output=lambda line: print(line, flush=True))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    summary = summarize(results)
    print("\n" + table(results))
    print(f"\nPassed {summary['passed']}/{summary['tasks']}; mean steps {summary['mean_steps']}; "
          f"tokens {summary['prompt_tokens']:,} prompt / {summary['generated_tokens']:,} generated; {summary['seconds']}s")
    report = {"model": MODEL, "started": started, "tasks_file": args.tasks,
              "settings": {"max_output_tokens": args.max_output_tokens, "prompt_budget": args.prompt_budget},
              "summary": summary, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Results written to {output}")
    return 0 if summary["passed"] == summary["tasks"] else 1


if __name__ == "__main__":
    sys.exit(main())
