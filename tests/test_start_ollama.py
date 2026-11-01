import contextlib
import io
import unittest
from unittest.mock import patch

from scripts.start_ollama import main


class LauncherTests(unittest.TestCase):
    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([])

    @patch("scripts.start_ollama.shutil.which", return_value=None)
    @patch("scripts.start_ollama.subprocess.run")
    def test_missing_binary(self, run, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()

    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection")
    @patch("scripts.start_ollama.subprocess.run")
    def test_existing_server_is_not_reused(self, run, connect, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()

    @patch.dict("os.environ", {"OLLAMA_NO_CLOUD": "0", "OLLAMA_HOST": "0.0.0.0:1234"})
    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection", side_effect=ConnectionRefusedError)
    @patch("scripts.start_ollama.subprocess.run")
    def test_overrides_settings_and_preserves_exit_status(self, run, connect, which):
        run.return_value.returncode = 7
        self.assertEqual(self.invoke(), 7)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/bin/ollama", "serve"])
        self.assertEqual(kwargs["env"]["OLLAMA_NO_CLOUD"], "1")
        self.assertEqual(kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:11434")

    @patch("scripts.start_ollama.shutil.which", return_value="/bin/ollama")
    @patch("scripts.start_ollama.socket.create_connection", side_effect=PermissionError)
    @patch("scripts.start_ollama.subprocess.run")
    def test_port_check_failure_does_not_start_server(self, run, connect, which):
        self.assertEqual(self.invoke(), 1)
        run.assert_not_called()
