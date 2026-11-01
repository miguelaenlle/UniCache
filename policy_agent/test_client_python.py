"""Boundary regressions using synthetic canaries, never credentials or trace data."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from policy_agent.client_python import install_client_python


@unittest.skipUnless(Path('/usr/bin/sandbox-exec').exists(), 'macOS Seatbelt required')
class ClientPythonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='unicache-client-boundary-')
        self.root = Path(self.temp.name).resolve()
        self.sandbox = self.root / 'sandbox'
        self.sandbox.mkdir()
        self.canary = self.root / 'DUMMY-CANARY.txt'
        self.canary.write_text('SYNTHETIC-CANARY-DO-NOT-READ')
        self.launcher = install_client_python(self.sandbox)

    def tearDown(self):
        self.temp.cleanup()

    def run_client(self, source, name='run_local.py', args=()):
        (self.sandbox / name).write_text(source)
        return subprocess.run([str(self.sandbox/'bin/python'), name, *args],
                              cwd=self.sandbox, text=True, capture_output=True,
                              timeout=15, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    def assert_canary_hidden(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(self.canary.read_text(), result.stdout + result.stderr)
        self.assertIn('DENIED', result.stdout)

    def test_modified_client_cannot_read_outside_dummy_canary(self):
        result = self.run_client(f'''from pathlib import Path
try:
    print(Path({str(self.canary)!r}).read_text())
except PermissionError:
    print("DENIED")
''')
        self.assert_canary_hidden(result)

    def test_cwd_json_shadow_cannot_run_before_sandbox(self):
        (self.sandbox/'json.py').write_text(f'''from pathlib import Path
try:
    print(Path({str(self.canary)!r}).read_text())
except PermissionError:
    print("DENIED")
''')
        # The trusted bootstrap imports standard-library json before adding cwd.
        # Also explicitly execute the malicious cwd module to prove confinement
        # remains effective if an editable client chooses to load it later.
        result = self.run_client('''import json, runpy
assert not json.__file__.endswith("sandbox/json.py")
runpy.run_path("json.py")
''')
        self.assert_canary_hidden(result)

    def test_normal_spool_write_rename_read_and_arguments(self):
        result = self.run_client('''import json, sys
from pathlib import Path
Path("requests").mkdir()
Path("feedback").mkdir()
pending = Path("requests/example.tmp")
pending.write_text(json.dumps({"arguments": sys.argv[1:]}))
pending.rename("requests/example.json")
response = Path("feedback/example.json")
response.write_text(Path("requests/example.json").read_text())
print(response.read_text())
''', args=('--rationale-file', 'a rationale.md'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'arguments':['--rationale-file','a rationale.md']})

    def test_actual_submit_smoke_and_local_clients_exchange_spool_messages(self):
        source_root = Path(__file__).resolve().parent
        for name in ('submit.py', 'smoke_test.py', 'run_local.py', 'local_client.py'):
            (self.sandbox/name).write_text((source_root/name).read_text())
        for directory in ('requests', 'feedback'):
            (self.sandbox/directory).mkdir()
        (self.sandbox/'policy.py').write_text('class Policy: pass\n')
        (self.sandbox/'rationale.md').write_text('Synthetic validation rationale.')
        for name, args, action in (('smoke_test.py', [], 'smoke_test'),
                                  ('run_local.py', ['test.py', 'argument'], 'local_test'),
                                  ('submit.py', ['--rationale-file', 'rationale.md'], None)):
            with self.subTest(client=name):
                received = []
                stop = threading.Event()
                def evaluator():
                    deadline = time.monotonic()+10
                    while time.monotonic()<deadline and not stop.is_set():
                        pending = list((self.sandbox/'requests').glob('*.json'))
                        if pending:
                            received.append(json.loads(pending[0].read_text()))
                            response = self.sandbox/'feedback'/pending[0].name
                            temporary = response.with_suffix('.tmp')
                            temporary.write_text(json.dumps({'ok':True, 'output':'SYNTHETIC OK'}))
                            temporary.rename(response)
                            return
                        time.sleep(.01)
                thread = threading.Thread(target=evaluator, daemon=True)
                thread.start()
                try:
                    result = subprocess.run([str(self.sandbox/'bin/python'), name, *args],
                                            cwd=self.sandbox, text=True, capture_output=True,
                                            timeout=12)
                finally:
                    stop.set()
                    thread.join(timeout=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('SYNTHETIC OK', result.stdout)
                self.assertEqual(len(received), 1)
                if action:
                    self.assertEqual(received[0]['action'], action)
                else:
                    self.assertEqual(received[0]['source'], 'class Policy: pass\n')
                    self.assertEqual(received[0]['rationale'], 'Synthetic validation rationale.')
                for directory in ('requests', 'feedback'):
                    for path in (self.sandbox/directory).glob('*'):
                        path.unlink()

    def test_cli_environment_and_private_home_are_not_inherited(self):
        home = self.sandbox/'home'
        home.mkdir()
        (home/'DUMMY-auth.txt').write_text('SYNTHETIC-HOME-CANARY')
        with patch.dict(os.environ, {'FAKE_SECRET':'DUMMY', 'ANTHROPIC_API_KEY':'DUMMY-API-KEY'}):
            result = self.run_client('''import os
from pathlib import Path
assert "FAKE_SECRET" not in os.environ
assert "ANTHROPIC_API_KEY" not in os.environ
assert os.environ["PATH"] == "/usr/bin:/bin"
assert Path.home() == Path.cwd()/"tmp"
for action in (lambda: Path("home/DUMMY-auth.txt").read_text(),
               lambda: Path("home/other.txt").write_text("bad"),
               lambda: os.listdir("home")):
    try:
        action()
    except PermissionError:
        print("DENIED")
    else:
        raise AssertionError("private CLI home unexpectedly accessible")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('DENIED'), 3)
        self.assertNotIn('SYNTHETIC-HOME-CANARY', result.stdout+result.stderr)
        self.assertEqual((home/'DUMMY-auth.txt').read_text(), 'SYNTHETIC-HOME-CANARY')

    def test_network_and_fork_are_denied(self):
        result = self.run_client('''import errno, socket, subprocess
for action in (lambda: socket.create_connection(("127.0.0.1", 9), timeout=1),
               lambda: subprocess.run(["/usr/bin/true"], check=True)):
    try:
        action()
    except OSError as error:
        assert error.errno in (errno.EPERM, errno.EACCES), repr(error)
        print("DENIED")
    else:
        raise AssertionError("sandbox unexpectedly allowed operation")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('DENIED'), 2)

    def test_bin_unlink_rename_chmod_and_replacement_are_denied(self):
        result = self.run_client('''import os
from pathlib import Path
for action in (lambda: Path("bin/python").unlink(),
               lambda: Path("bin").rename("moved-bin"),
               lambda: os.chmod("bin", 0o777),
               lambda: Path("bin/replacement").write_text("bad")):
    try:
        action()
    except PermissionError:
        print("DENIED")
    else:
        raise AssertionError("sandbox unexpectedly allowed entry-point mutation")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('DENIED'), 4)
        self.assertEqual((self.sandbox/'bin/python').resolve(), self.launcher)

    def test_outside_script_and_python_command_flags_are_rejected(self):
        outside = self.root/'outside.py'
        outside.write_text('raise AssertionError("outside script executed")')
        for args in ((str(outside),), ('-c', 'print("unconfined")')):
            result = subprocess.run([str(self.sandbox/'bin/python'), *args], cwd=self.sandbox,
                                    text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('inside the sandbox', result.stderr)
            self.assertNotIn('unconfined', result.stdout)


if __name__ == '__main__':
    unittest.main()
