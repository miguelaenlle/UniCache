"""Offline integration tests for local Python execution; no Claude calls."""
from pathlib import Path
import shutil
import json
import subprocess
import sys
import threading
import time
import tempfile
import unittest
from policy_agent.local_runner import run_test


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='local-tests-', dir='result')
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root/'sandbox'
        self.workspace.mkdir()
        (self.workspace/'home').mkdir()
        shutil.copyfile('policy_agent/starter.py',self.workspace/'policy.py')

    def tearDown(self):
        self.temp.cleanup()

    def script(self, source, **kwargs):
        (self.workspace/'check.py').write_text(source)
        return run_test(self.workspace, {'action':'local_test','script':'check.py','args':['hello']}, **kwargs)

    def test_smoke_pass_and_invalid_choice(self):
        result = run_test(self.workspace,{'action':'smoke_test'})
        self.assertTrue(result['ok'], result['output'])
        self.assertIn('No trace data',result['output'])
        p=self.workspace/'policy.py'
        p.write_text(p.read_text().replace('return None','return -1'))
        result = run_test(self.workspace,{'action':'smoke_test'})
        self.assertFalse(result['ok'])
        self.assertIn('InvalidEvictionError',result['output'])
        self.assertIn('Traceback',result['output'])

    def test_custom_import_args_and_scratch_writes(self):
        result = self.script("from policy import Policy\nimport sys\nfrom pathlib import Path\nPolicy()\nPath('local-output/proof.txt').write_text(sys.argv[1])\nprint('custom test passed')\n")
        self.assertTrue(result['ok'],result['output'])
        self.assertEqual((self.workspace/'local-output/proof.txt').read_text(),'hello')

    def test_hidden_file_and_auth_denied(self):
        hidden=self.root/'private-trace.txt'
        hidden.write_text('trace sentinel')
        result=self.script(f'open({str(hidden)!r}).read()')
        self.assertFalse(result['ok'])
        self.assertIn('PermissionError',result['output'])
        (self.workspace/'home/credentials').write_text('credential sentinel')
        result=self.script("open('home/credentials').read()")
        self.assertFalse(result['ok'])
        self.assertIn('PermissionError',result['output'])

    def test_network_and_workspace_writes_denied(self):
        result=self.script("import socket\nsocket.create_connection(('127.0.0.1',80))")
        self.assertFalse(result['ok'])
        self.assertIn('PermissionError',result['output'])
        result=self.script("open('policy.py','w').write('changed')")
        self.assertFalse(result['ok'])
        self.assertIn('PermissionError',result['output'])

    def test_limits(self):
        result=self.script('while True: pass',timeout=.3)
        self.assertFalse(result['ok'])
        self.assertIn('timed out',result['output'])
        result=self.script("print('x'*1100000)")
        self.assertFalse(result['ok'])
        self.assertIn('exceeded 1 MiB',result['output'])

    def test_client_round_trip_leaves_no_test_logs(self):
        for name in ['requests','feedback']:
            (self.workspace/name).mkdir()
        for name in ['local_client.py','run_local.py','smoke_test.py']:
            shutil.copyfile(Path('policy_agent')/name,self.workspace/name)
        (self.workspace/'check.py').write_text("print('client test passed')")
        (self.workspace/'history.jsonl').write_text('submission history sentinel')
        failures=[]
        def broker():
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    pending=list((self.workspace/'requests').glob('*.json'))
                    if pending:
                        request=pending[0]
                        result=run_test(self.workspace,json.loads(request.read_text()))
                        path=self.workspace/'feedback'/request.name
                        tmp=path.with_suffix('.tmp')
                        tmp.write_text(json.dumps(result))
                        tmp.replace(path)
                        return
                    time.sleep(.02)
                raise TimeoutError('Client did not submit its local test request')
            except Exception as error:
                failures.append(error)
        thread=threading.Thread(target=broker,daemon=True)
        thread.start()
        try:
            result=subprocess.run([sys.executable,'run_local.py','check.py'],cwd=self.workspace,
                                  capture_output=True,text=True,timeout=8)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            self.assertIn('client test passed',result.stdout)
        finally:
            thread.join(timeout=5)
        self.assertFalse(failures,failures)
        self.assertEqual((self.workspace/'history.jsonl').read_text(),'submission history sentinel')
        self.assertEqual(list((self.workspace/'requests').iterdir()),[])
        self.assertEqual(list((self.workspace/'feedback').iterdir()),[])

    def test_path_escape_rejected(self):
        outside=self.root/'outside.py'
        outside.write_text('print(123)')
        with self.assertRaises(ValueError):
            run_test(self.workspace,{'action':'local_test','script':'../outside.py'})


if __name__ == '__main__':
    unittest.main()
