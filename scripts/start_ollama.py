#!/usr/bin/env python3
"""Start a foreground, loopback-only Ollama server with cloud disabled."""
import argparse
import os
import shutil
import socket
import subprocess
import sys


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    executable = shutil.which("ollama")
    if executable is None:
        print("Error: Ollama is not installed or is missing from PATH. Install it from https://ollama.com/.",
              file=sys.stderr)
        return 1
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            print("Error: Port 11434 is already in use. Quit the Ollama app or stop its service, "
                  "then run this script again. An existing server cannot inherit these settings.",
                  file=sys.stderr)
            return 1
    except ConnectionRefusedError:
        pass
    except OSError as exc:
        print(f"Error: Cannot check the local Ollama port: {exc}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["OLLAMA_NO_CLOUD"] = "1"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    print("Starting Ollama with cloud disabled on 127.0.0.1:11434. Press Ctrl+C to stop.\n"
          "Check the server logs for: Ollama cloud disabled: true", flush=True)
    try:
        return subprocess.run([executable, "serve"], env=env, check=False).returncode
    except OSError as exc:
        print(f"Error: Could not start Ollama: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
