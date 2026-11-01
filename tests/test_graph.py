"""Behavioral coverage for the LangGraph migration."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langsmith import get_tracing_context

from offline_agent.agent import build_graph, run
from offline_agent.cli import main
from offline_agent.workspace import Workspace


def action(kind, path="", content=""):
    return dict(action=kind, path=path, content=content, message=kind)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)

    def test_read_result_reaches_next_model_call(self):
        Path(self.temp.name, "code.py").write_text("value = 42\n")
        calls = []

        class Model:
            def chat(self, messages, schema):
                calls.append(messages)
                return action("read", "code.py") if len(calls) == 1 else action("finish")

        self.assertEqual(run(Model(), self.ws, "Inspect code", output=lambda _: None), 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("value = 42", calls[1][-1]["content"])
        self.assertEqual(calls[1][-2]["role"], "assistant")

    def test_thirty_model_actions_stop_without_recursion_error(self):
        class Model:
            calls = 0

            def chat(self, messages, schema):
                self.calls += 1
                return action("read", "missing.py")

        model = Model()
        self.assertEqual(run(model, self.ws, "Task", max_steps=30, output=lambda _: None), 2)
        self.assertEqual(model.calls, 30)

    def test_finish_on_last_allowed_action_succeeds(self):
        class Model:
            def chat(self, messages, schema):
                return action("finish")

        self.assertEqual(run(Model(), self.ws, "Task", max_steps=1, output=lambda _: None), 0)

    def test_approved_write_on_last_action_is_applied(self):
        class Model:
            def chat(self, messages, schema):
                return action("write", "new.py", "answer = 42\n")

        self.assertEqual(run(Model(), self.ws, "Task", max_steps=1, apply=True,
                             confirm=lambda _: True, output=lambda _: None), 2)
        self.assertEqual(self.ws.read("new.py"), "answer = 42\n")

    @patch.dict("os.environ", {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"})
    def test_parent_tracing_is_disabled_for_model_execution(self):
        observed = []

        class Model:
            def chat(self, messages, schema):
                observed.append(get_tracing_context()["enabled"])
                return action("finish")

        run(Model(), self.ws, "Task", output=lambda _: None)
        self.assertEqual(observed, [False])

    def test_invalid_limit_rejected_before_model_execution(self):
        for limit in (0, 31):
            with self.assertRaises(ValueError):
                build_graph(None, self.ws, max_steps=limit)

    def test_cli_doctor_uses_fixed_model(self):
        with patch("offline_agent.cli.Ollama") as model, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--doctor"]), 0)
        model.assert_called_once_with(endpoint="http://127.0.0.1:11434")
        model.return_value.check_model.assert_called_once_with()
