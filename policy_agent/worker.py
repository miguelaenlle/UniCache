"""Standalone sandboxed policy worker. No simulator imports or trace access."""
import contextlib
import io
import json
import linecache
import resource
import sys
import traceback

resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
# Seatbelt enforces filesystem/network isolation; each RPC has a host deadline.
namespace = {'__name__': 'candidate'}
try:
    source = json.loads(sys.stdin.readline())['source']
    linecache.cache['policy.py'] = (len(source), None, source.splitlines(True), 'policy.py')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, 'policy.py', 'exec'), namespace)
        policy = namespace['Policy']()
        if not all(callable(getattr(policy, name, None)) for name in ('on_event', 'evict')):
            raise TypeError('Policy must implement on_event(event) and evict(now)')
    print(json.dumps({'ok': True}), flush=True)
except BaseException as error:
    print(json.dumps({'error': type(error).__name__ + ': ' + str(error),
                      'traceback': traceback.format_exc()}), flush=True)
    sys.exit(1)
for line in sys.stdin:
    try:
        message = json.loads(line)
        with contextlib.redirect_stdout(io.StringIO()):
            for event in message['events']:
                policy.on_event(event)
            result = policy.evict(message['now']) if message['choose'] else None
        print(json.dumps({'result': result}), flush=True)
    except BaseException as error:
        print(json.dumps({'error': type(error).__name__ + ': ' + str(error),
                          'traceback': traceback.format_exc()}), flush=True)
