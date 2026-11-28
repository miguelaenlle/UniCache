"""Build offline Python sandboxes with Seatbelt or Bubblewrap."""

from pathlib import Path
import shutil
import sys
import sysconfig


def sandbox_prefix(read_paths, *, rules, workspace=None, writable=None,
                   read_only=(), hidden=()):
    if sys.platform == 'darwin' and Path('/usr/bin/sandbox-exec').is_file():
        return ['/usr/bin/sandbox-exec', '-p', rules]
    bubblewrap = shutil.which('bwrap', path='/usr/bin:/bin')
    if sys.platform != 'linux' or not bubblewrap:
        raise RuntimeError('Policy isolation requires Seatbelt on macOS or Bubblewrap on Linux')

    command = [
        bubblewrap, '--unshare-all', '--unshare-user', '--disable-userns', '--cap-drop', 'ALL',
        '--die-with-parent', '--new-session', '--dev', '/dev',
    ]
    # Python and its extension modules need the host dynamic loader and libraries.
    for path in map(Path, ('/usr/lib', '/usr/lib64', '/lib', '/lib64')):
        if path.is_symlink():
            command.extend(['--symlink', str(path.readlink()), str(path)])
        elif path.is_dir():
            command.extend(['--ro-bind', str(path), str(path)])

    runtime = Path(sys.executable).resolve()
    paths = [
        runtime, sysconfig.get_path('stdlib'),
        *runtime.parent.parent.glob('lib/libpython*.so*'),
        *read_paths,
    ]
    for path in dict.fromkeys(map(str, paths)):
        command.extend(['--ro-bind', path, path])

    if workspace is not None:
        command.extend(['--ro-bind', str(workspace), str(workspace)])
    if writable is not None:
        command.extend(['--bind', str(writable), str(writable)])
    for path in read_only:
        command.extend(['--ro-bind', str(path), str(path)])
    for path in hidden:
        command.extend(['--tmpfs', str(path), '--remount-ro', str(path)])
    command.extend(['--chdir', str(workspace or '/'), '--'])
    return command


# Install this filter inside the namespace before importing editable Python.
# Namespace isolation prevents host network access. The filter also blocks new
# sockets and child processes, including when the workspace contains UNIX sockets.
_LINUX_BOOTSTRAP = """\
def _restrict():
    import ctypes, errno
    seccomp = ctypes.CDLL('libseccomp.so.2')
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError('Could not initialize seccomp')
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO
        for name in (b'socket', b'socketpair', b'fork', b'vfork', b'clone', b'clone3'):
            number = seccomp.seccomp_syscall_resolve_name(name)
            if number != -1 and seccomp.seccomp_rule_add(context, deny, number, 0):
                raise RuntimeError(f'Could not block {name!r}')
        if seccomp.seccomp_load(context):
            raise RuntimeError('Could not activate seccomp')
    finally:
        seccomp.seccomp_release(context)

_restrict()
del _restrict
"""


def python_source(source):
    if sys.platform == 'linux':
        # Compile separately to preserve source line numbers and future imports.
        return _LINUX_BOOTSTRAP + f'exec(compile({source!r}, "<sandbox>", "exec"))'
    return source
