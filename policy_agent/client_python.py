"""Install a trusted offline Python entry point for editable sandbox clients.

The launcher lives outside the coding workspace. Neither the client's source nor a
cwd module runs before Seatbelt is active. The runner must also prevent its coding
CLI's Write/Edit tools from replacing sandbox/bin/python or its parent directory.
"""
import json
from pathlib import Path
import shlex
import sys

from policy_agent.bridge import profile


_INNER = '''import json, os, runpy, sys
workspace, script, *arguments = sys.argv[1:]
os.chdir(workspace)
sys.path.insert(0, workspace)
sys.argv = [script, *arguments]
runpy.run_path(script, run_name="__main__")
'''


def install_client_python(sandbox, python=None):
    """Replace bin/python with a launcher outside `sandbox`; return its path.

    Safe to call again when upgrading an existing paused run. Files under the
    workspace remain editable; changing them cannot change the outer bootstrap.
    """
    sandbox = Path(sandbox).resolve()
    runtime = Path(python or sys.executable).resolve()
    internal = sandbox.parent / 'internal'
    internal.mkdir(parents=True, exist_ok=True)
    binary_dir = sandbox / 'bin'
    binary_dir.mkdir(parents=True, exist_ok=True)
    temporary = sandbox / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    launcher = internal / 'client-python.sh'
    rules = profile([sys.base_prefix, runtime], writable=sandbox)
    # Deny writing the entry point even from a modified Python client. This covers
    # unlink, rename and attribute changes, not just opening a file for writing.
    rules += '\n(deny file-write* (subpath ' + json.dumps(str(binary_dir)) + '))'
    private_home = json.dumps(str(sandbox / 'home'))
    rules += '\n(deny file-read-data (subpath ' + private_home + '))'
    rules += '\n(deny file-write* (subpath ' + private_home + '))'
    clean_env = {'PATH': '/usr/bin:/bin', 'HOME': str(temporary),
                 'TMPDIR': str(temporary), 'PYTHONDONTWRITEBYTECODE': '1'}
    bootstrap = f'''import os, sys
from pathlib import Path
workspace = Path({str(sandbox)!r})
if len(sys.argv) < 2:
    raise SystemExit("Usage: python script.py [arguments]")
script = Path(sys.argv[1])
script = (workspace / script).resolve() if not script.is_absolute() else script.resolve()
if not script.is_relative_to(workspace) or script.suffix != ".py":
    raise SystemExit("Client script must be a Python file inside the sandbox")
os.execve("/usr/bin/sandbox-exec", ["sandbox-exec", "-p", {rules!r},
    {str(runtime)!r}, "-I", "-S", "-c", {_INNER!r}, str(workspace), str(script), *sys.argv[2:]], {clean_env!r})
'''
    command = '#!/bin/sh\nexec ' + shlex.quote(str(runtime)) + ' -I -S -c ' + shlex.quote(bootstrap) + ' "$@"\n'
    if launcher.exists():
        launcher.chmod(0o700)
    launcher.write_text(command)
    launcher.chmod(0o555)
    entry = binary_dir / 'python'
    if entry.is_symlink() or entry.exists():
        entry.unlink()
    entry.symlink_to(launcher)
    return launcher
