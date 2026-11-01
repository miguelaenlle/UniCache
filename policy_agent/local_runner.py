"""Trusted local-test execution. Never run agent scripts in the evaluator process."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from policy_agent.bridge import profile

MAX_OUTPUT = 1024*1024
BOOTSTRAP = '''
import os, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
workspace, script, *args = sys.argv[1:]
os.chdir(workspace)
sys.path.insert(0, workspace)
sys.argv = [script, *args]
runpy.run_path(script, run_name='__main__')
'''


def run_test(workspace, message, timeout=30):
    workspace = Path(workspace).resolve()
    action = message['action']
    if action == 'smoke_test':
        script = Path(__file__).with_name('smoke_suite.py').resolve()
        args = []
    elif action == 'local_test':
        script = (workspace/message['script']).resolve()
        if not script.is_relative_to(workspace) or script.suffix != '.py' or not script.is_file():
            raise ValueError('Script must be a .py file inside the sandbox')
        args = message.get('args', [])
        if not isinstance(args,list) or any(not isinstance(x,str) for x in args):
            raise ValueError('Script arguments must be strings')
    else:
        raise ValueError('Unknown local-test action')
    output = workspace/'local-output'
    output.mkdir(exist_ok=True)
    if output.is_symlink() or not output.resolve().is_relative_to(workspace):
        raise ValueError('Unsafe local-output path')
    runtime = Path(sys.executable).resolve()
    rules = profile([workspace, script, sys.base_prefix, runtime], writable=output, network=False)
    # Coding-session auth/config is not an input to local Python tests.
    rules += '\n(deny file-read-data (subpath ' + json.dumps(str(workspace/'home')) + '))'
    command = ['/usr/bin/sandbox-exec','-p',rules,str(runtime),'-I','-S','-c',BOOTSTRAP,
               str(workspace),str(script),*args]
    env = {'PATH':'/usr/bin:/bin', 'HOME':str(output), 'TMPDIR':str(output)}
    process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    collected = bytearray()
    problem = None
    start = time.monotonic()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ)
    try:
        while selector.get_map():
            if time.monotonic()-start > timeout:
                problem = 'Local test timed out.'
                break
            for key, _ in selector.select(.1):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                collected.extend(data)
                if len(collected) > MAX_OUTPUT:
                    problem = 'Local test output exceeded 1 MiB.'
                    break
            if problem:
                break
    finally:
        selector.close()
        if not problem:
            try:
                process.wait(timeout=max(.1, timeout-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                problem = 'Local test timed out.'
        if process.poll() is None:
            process.kill()
        status = process.wait()
        process.stdout.close()
    text = collected[:MAX_OUTPUT].decode('utf-8',errors='replace')
    if problem:
        text += '\n'+problem+'\n'
    text += f'\nLocal test: {"OK" if status == 0 and not problem else "FAILED"} (exit {status})\n'
    return {'ok':status == 0 and not problem,'output':text}
