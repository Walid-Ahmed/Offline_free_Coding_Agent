import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline_agent.agent import run
from offline_agent.test_runner import run_tests
from offline_agent.workspace import Workspace


@unittest.skipUnless(os.name == "posix", "Test runner supports macOS/Linux")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tests").mkdir()
        self.ws = Workspace(self.root)

    def script(self, code):
        (self.root / "tests" / "test_example.py").write_text(code)

    def test_passing_tests(self):
        self.script("import unittest\nclass Test(unittest.TestCase):\n def test_add(self): self.assertEqual(2+2, 4)\n")
        result = run_tests(self.ws)
        self.assertIn("Exit code: 0", result)
        self.assertIn("Ran 1 test", result)
        self.assertIn("OK", result)

    def test_failure_output_and_exit_code(self):
        self.script("import unittest\nclass Test(unittest.TestCase):\n def test_fail(self): self.fail('expected failure')\n")
        result = run_tests(self.ws)
        self.assertIn("Exit code: 1", result)
        self.assertIn("expected failure", result)

    def test_timeout(self):
        self.script("import time\ntime.sleep(60)\n")
        self.assertIn("Timed out", run_tests(self.ws, timeout=0.3))

    def test_output_is_bounded(self):
        self.script("print('x' * 100000)\n")
        result = run_tests(self.ws, output_limit=100)
        self.assertIn("Output truncated", result)
        self.assertLess(len(result), 200)

    def test_missing_tests_directory(self):
        (self.root / "tests").rmdir()
        with self.assertRaises(ValueError):
            run_tests(self.ws)


class GraphRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)

    def model(self, path="", content=""):
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return {"action": "run_tests" if len(calls) == 1 else "finish",
                        "path": path if len(calls) == 1 else "", "content": content,
                        "message": "Request tests"}

        return Model(), calls

    @patch("offline_agent.agent.run_tests")
    def test_no_approval_never_executes(self, runner):
        for consent in (None, lambda: False):
            model, calls = self.model()
            run(model, self.ws, "Test", confirm_tests=consent, output=lambda _: None)
            self.assertIn("NOT run", calls[1][-1]["content"])
        runner.assert_not_called()

    @patch("offline_agent.agent.run_tests", return_value="Exit code: 1\nFAILED")
    def test_approved_result_reaches_model_without_apply(self, runner):
        model, calls = self.model()
        run(model, self.ws, "Test", confirm_tests=lambda: True, output=lambda _: None)
        runner.assert_called_once_with(self.ws)
        self.assertIn("Exit code: 1\nFAILED", calls[1][-1]["content"])

    @patch("offline_agent.agent.run_tests")
    def test_custom_command_rejected(self, runner):
        model, calls = self.model(content="rm -rf anything")
        run(model, self.ws, "Test", confirm_tests=lambda: True, output=lambda _: None)
        runner.assert_not_called()
        self.assertIn("no path, content, or custom command", calls[1][-1]["content"])
