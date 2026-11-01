"""Opt-in real container checks; never replaced with host execution."""
import os
import tempfile
import shutil
import subprocess
import uuid
from unittest.mock import patch
from types import SimpleNamespace
import unittest
from pathlib import Path

from offline_agent.test_runner import run_tests, local_socket
from offline_agent.workspace import Workspace


@unittest.skipUnless(os.environ.get('OFFLINE_AGENT_SANDBOX_INTEGRATION') == '1',
                     'Set OFFLINE_AGENT_SANDBOX_INTEGRATION=1 with Docker/image ready')
class SandboxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tests').mkdir()
        self.ws = Workspace(self.root)

    def script(self, text):
        (self.root / 'tests/test_example.py').write_text(text)

    def test_unittest_pass_and_fail(self):
        self.script('import unittest\nclass Test(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2, 4)\n')
        self.assertIn('Exit code: 0', run_tests(self.ws))
        self.script('import unittest\nclass Test(unittest.TestCase):\n def test_no(self): self.fail("expected failure")\n')
        result = run_tests(self.ws)
        self.assertIn('Exit code: 1', result)
        self.assertIn('expected failure', result)

    def test_pytest_pass_and_fail(self):
        self.script('def test_ok():\n assert 2+2 == 4\n')
        result = run_tests(self.ws, runner='pytest')
        self.assertIn('Exit code: 0', result)
        self.assertIn('1 passed', result)
        self.script('def test_no():\n assert 2+2 == 5\n')
        self.assertIn('Exit code: 1', run_tests(self.ws, runner='pytest'))

    def test_network_filesystem_and_environment_restrictions(self):
        with tempfile.TemporaryDirectory(prefix='host-canary-') as outside:
            canary = Path(outside, 'secret.txt')
            canary.write_text('host-only-data')
            (self.root / '.gitignore').write_text('private/\n')
            (self.root / 'private').mkdir()
            (self.root / 'private/secret.txt').write_text('excluded')
            self.script('''import os, socket, fcntl, struct
from pathlib import Path
import pytest

def test_boundaries():
    assert os.getuid() == 65534
    mounts = {parts[1]: set(parts[3].split(',')) for parts in (line.split() for line in Path('/proc/mounts').read_text().splitlines())}
    assert 'ro' in mounts['/workspace']
    assert 'ro' in mounts['/']
    active = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, name in socket.if_nameindex():
            flags = fcntl.ioctl(probe.fileno(), 0x8913, struct.pack('256s', name.encode()))
            if struct.unpack_from('H', flags, 16)[0] & 1:
                active.add(name)
    assert active == {'lo'}
    status = Path('/proc/self/status').read_text()
    assert 'CapEff:\t0000000000000000' in status
    assert 'NoNewPrivs:\t1' in status
    with pytest.raises(OSError):
        socket.create_connection(('198.51.100.1', 443), timeout=0.5)
    assert not Path(CANARY).exists()
    assert not Path('/workspace/private/secret.txt').exists()
    assert not Path('/var/run/docker.sock').exists()
    assert os.getenv('OFFLINE_TEST_HOST_SECRET') is None
    for name in ('/workspace/should-not-write', '/should-not-write'):
        with pytest.raises(OSError):
            Path(name).write_text('blocked')
    Path('/tmp/permitted').write_text('temporary')
    assert Path('/tmp/permitted').read_text() == 'temporary'
'''.replace('CANARY', repr(str(canary))))
            original = os.environ.get('OFFLINE_TEST_HOST_SECRET')
            os.environ['OFFLINE_TEST_HOST_SECRET'] = 'must-not-leak'
            try:
                result = run_tests(Workspace(self.root), runner='pytest')
            finally:
                if original is None:
                    os.environ.pop('OFFLINE_TEST_HOST_SECRET', None)
                else:
                    os.environ['OFFLINE_TEST_HOST_SECRET'] = original
            self.assertIn('Exit code: 0', result, result)
            self.assertEqual(canary.read_text(), 'host-only-data')
            self.assertFalse((self.root / 'should-not-write').exists())

    def test_timeout_and_bounded_output(self):
        self.script('import time\ntime.sleep(60)\n')
        self.assertIn('Timed out', run_tests(self.ws, timeout=2))
        self.script('print("x" * 100000)\ndef test_ok(): pass\n')
        self.assertIn('Output truncated', run_tests(self.ws, output_limit=100))

    def test_timeout_removes_container_and_detached_children(self):
        self.script("import unittest, subprocess, sys, time\nclass Test(unittest.TestCase):\n def test_child(self):\n  subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n  time.sleep(60)\n")
        token = uuid.uuid4().hex
        with patch('offline_agent.test_runner.uuid.uuid4', return_value=SimpleNamespace(hex=token)):
            self.assertIn('Timed out', run_tests(self.ws, timeout=2))
        name = 'offline-agent-tests-' + token
        result = subprocess.run([shutil.which('docker'), '--host', 'unix://' + str(local_socket()),
                                 'ps', '-a', '--filter', 'name=' + name, '--format', '{{.Names}}'],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout.strip(), b'')
