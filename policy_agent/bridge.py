"""Trusted adapter; submitted Python runs only in the Seatbelt worker."""
import enum
import inspect
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def profile(read_paths, writable=None, network=False):
    def q(s):
        return json.dumps(str(s))
    paths = ['/System', '/usr', '/Library', '/private/etc', '/etc', '/private/var/db/dyld', '/opt/homebrew', '/dev', *read_paths]
    if writable:
        paths.append(writable)
    exceptions = ' '.join(f'(require-not (subpath {q(path)}))' for path in paths)
    rules = ['(version 1)', '(allow default)',
             f'(deny file-read-data (require-all (require-not (vnode-type DIRECTORY)) {exceptions}))']
    write_exceptions = '(require-not (subpath "/dev"))'
    if writable:
        write_exceptions += f' (require-not (subpath {q(writable)}))'
    rules.append(f'(deny file-write* (require-all {write_exceptions}))')
    if not network:
        rules.extend(['(deny network*)', '(deny process-fork)'])
    return '\n'.join(rules)


class PolicyFailure(RuntimeError):
    pass


class Remote:
    def __init__(self, source):
        worker = Path(__file__).with_name('worker.py').resolve()
        self.process = subprocess.Popen(
            ['/usr/bin/sandbox-exec', '-p', profile([worker, sys.base_prefix, Path(sys.executable).resolve()]),
             str(Path(sys.executable).resolve()), '-I', '-S', '-c', worker.read_text()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env={'PATH': '/usr/bin:/bin'})
        try:
            self.call({'source': source})
        except BaseException:
            self.close()
            raise

    def call(self, data):
        try:
            self.process.stdin.write(json.dumps(data) + '\n')
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], 30)
            if not ready:
                self.process.kill()
                raise PolicyFailure('PolicyTimeout: no response within 30 seconds')
            line = self.process.stdout.readline(4 * 1024 * 1024)
            if not line:
                raise PolicyFailure('Policy worker startup/exit error: ' + self.process.stderr.read(8192))
            result = json.loads(line)
            if 'error' in result:
                raise PolicyFailure(result['error'] + '\n' + result.get('traceback', ''))
            return result.get('result')
        except (OSError, ValueError) as error:
            raise PolicyFailure('Policy worker exited or returned invalid JSON') from error

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


def scalar(value):
    if isinstance(value, enum.Enum):
        return value.name
    return value


class ControlledEvictor:
    def __init__(self, base, source):
        self.base = base
        self.remote = Remote(source)
        self.eligible = {}
        self.events = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __contains__(self, block_id):
        return block_id in self.base

    def emit(self, event):
        self.events.append(event)
        if len(self.events) >= 2048:
            self.flush(False)

    def flush(self, choose):
        from vllm.simulator.timer_utils import GLOBAL_TIMER
        result = self.remote.call({'events': self.events, 'choose': choose,
                                   'now': GLOBAL_TIMER.get_current_time()})
        self.events = []
        return result

    def add(self, *args, **kwargs):
        data = dict(inspect.signature(self.base.add).bind(*args, **kwargs).arguments)
        self.base.add(*args, **kwargs)
        block_id = data['block_id']
        self.eligible[block_id] = data['content_hash']
        # Content hashes are opaque identities, not future-use information.
        self.emit({'kind': 'add', **{k: scalar(v) for k, v in data.items()}})

    def remove(self, block_id):
        self.base.remove(block_id)
        self.eligible.pop(block_id, None)
        self.emit({'kind': 'remove', 'block_id': block_id})

    def update(self, *args, **kwargs):
        data = dict(inspect.signature(self.base.update).bind(*args, **kwargs).arguments)
        self.base.update(*args, **kwargs)
        self.emit({'kind': 'update', **{k: scalar(v) for k, v in data.items()}})

    def evict(self):
        block_id = self.flush(True)
        if block_id is None:
            block_id, content_hash = self.base.evict()
            if block_id not in self.eligible or self.eligible[block_id] != content_hash:
                raise PolicyFailure('InvalidEvictionError: starting policy returned an ineligible block')
        else:
            if type(block_id) is not int or block_id not in self.eligible or block_id not in self.base:
                raise PolicyFailure(f'InvalidEvictionError: Block {block_id!r} is not eligible for eviction')
            content_hash = self.eligible[block_id]
            self.base.remove(block_id)
        self.eligible.pop(block_id)
        self.emit({'kind': 'remove', 'block_id': block_id})
        return block_id, content_hash
