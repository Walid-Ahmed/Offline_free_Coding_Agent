"""Approved test execution for trusted projects; this is not a sandbox."""
import os
import signal
import subprocess
import sys
import threading

TIMEOUT = 30
OUTPUT_LIMIT = 16_000


def command():
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]


def run_tests(workspace, *, timeout=TIMEOUT, output_limit=OUTPUT_LIMIT):
    """Run fixed unittest discovery, bounding duration and captured output.

    POSIX process groups allow cleanup of ordinary test subprocesses. Project
    code still executes with the user's permissions, including network access.
    """
    if os.name != "posix":
        raise ValueError("The test runner currently supports macOS and Linux only")
    tests = workspace.path("tests")
    if not tests.is_dir():
        raise ValueError("A tests/ directory is required for unittest discovery")
    captured = bytearray()
    truncated = threading.Event()
    with subprocess.Popen(command(), cwd=workspace.root, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        def collect():
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                remaining = output_limit - len(captured)
                captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated.set()

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        timed_out = False
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            # Also clean up children left behind after the test process exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join(timeout=2)
        text = bytes(captured).decode("utf-8", errors="replace")
        status = f"Timed out after {timeout} seconds" if timed_out else f"Exit code: {process.returncode}"
        if truncated.is_set():
            text += f"\n[Output truncated to {output_limit} bytes]"
        return status + "\n" + text
