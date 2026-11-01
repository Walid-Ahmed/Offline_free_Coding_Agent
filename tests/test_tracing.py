import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langsmith import get_tracing_context

from offline_agent.agent import run
from offline_agent.cli import main
from offline_agent.workspace import Workspace


class TracingTests(unittest.TestCase):
    def test_model_error_still_flushes_traces_without_masking_original_error(self):
        class Model:
            def chat(self, messages, schema):
                raise ValueError("Model failed")
        class FakeTracer(BaseCallbackHandler):
            def __init__(self, **kwargs):
                pass
        client = Mock()
        client.flush.side_effect = RuntimeError("Trace transport failed")
        with tempfile.TemporaryDirectory() as root, patch(
            "langchain_core.tracers.langchain.LangChainTracer", FakeTracer
        ), self.assertRaisesRegex(ValueError, "Model failed"):
            run(Model(), Workspace(root), "Explain", tracing_client=client, output=lambda _: None)
        client.flush.assert_called_once()

    def test_opt_in_enables_context_and_attaches_explicit_client(self):
        observed = []
        class Model:
            def chat(self, messages, schema):
                observed.append(get_tracing_context()["enabled"])
                return dict(action="finish", path="", content="Done", message="Done")
        client = Mock()
        constructors = []
        class FakeTracer(BaseCallbackHandler):
            def __init__(self, **kwargs):
                constructors.append(kwargs)

        with tempfile.TemporaryDirectory() as root, patch(
            "langchain_core.tracers.langchain.LangChainTracer", FakeTracer
        ):
            run(Model(), Workspace(root), "Explain", tracing_client=client,
                tracing_project="private-debug", output=lambda _: None)
        self.assertEqual(observed, [True])
        self.assertEqual(constructors, [dict(client=client, project_name="private-debug")])
        client.flush.assert_called_once()

    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_cli_redacts_content_by_default_and_forwards_opt_in(self):
        with tempfile.TemporaryDirectory() as root, patch("langsmith.Client") as client, \
             patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--workspace", root, "--langsmith-tracing", "Explain"]), 0)
        client.assert_called_once_with(api_key="test-key", hide_inputs=True, hide_outputs=True)
        self.assertIs(agent.call_args.kwargs["tracing_client"], client.return_value)

    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_TRACING": "true"})
    def test_environment_alone_does_not_construct_tracing_client(self):
        with tempfile.TemporaryDirectory() as root, patch("langsmith.Client") as client, \
             patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0) as agent, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["--workspace", root, "Explain"])
        client.assert_not_called()
        self.assertIsNone(agent.call_args.kwargs["tracing_client"])

    @patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"})
    def test_content_export_requires_separate_flag(self):
        with tempfile.TemporaryDirectory() as root, patch("langsmith.Client") as client, \
             patch("offline_agent.cli.Ollama"), patch("offline_agent.cli.run", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()):
            main(["--workspace", root, "--langsmith-tracing", "--langsmith-include-content", "Explain"])
        client.assert_called_once_with(api_key="test-key", hide_inputs=False, hide_outputs=False)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_key_fails_before_model_access(self):
        with patch("offline_agent.cli.Ollama") as model, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--langsmith-tracing", "Explain"]), 1)
        model.assert_not_called()
