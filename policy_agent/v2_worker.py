"""Isolated v2 policy runtime. Only source, config and public events enter here."""
import contextlib
import io
import json
import linecache
import resource
import sys
import traceback

resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def response(value):
    print(json.dumps(value, allow_nan=False), flush=True)


try:
    init = json.loads(sys.stdin.readline())
    for path in init.get('dependency_paths', []):
        sys.path.append(path)
    source = init['source']
    linecache.cache['policy.py'] = (len(source), None, source.splitlines(True), 'policy.py')
    namespace = {'__name__': 'candidate'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, 'policy.py', 'exec'), namespace)
        policy = namespace['Policy'](init['config'])
        if not all(callable(getattr(policy, name, None)) for name in ('on_event', 'evict')):
            raise TypeError('Policy must implement on_event(event) and evict(context)')
        mode = getattr(policy, 'CACHE_MODE', 'prefix')
        if mode not in ('prefix', 'disabled'):
            raise ValueError('CACHE_MODE must be prefix or disabled')
    response({'result': {'cache_mode': mode}})
except BaseException as error:
    response({'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
    sys.exit(1)

last_event_id = 0
for line in sys.stdin:
    try:
        message = json.loads(line)
        with contextlib.redirect_stdout(io.StringIO()):
            for event in message.get('events', []):
                if event['event_id'] <= last_event_id:
                    raise ValueError('Event IDs must increase')
                policy.on_event(event)
                last_event_id = event['event_id']
            result = None
            if message.get('choose'):
                context = message['context']
                if context['through_event_id'] != last_event_id:
                    raise ValueError('Eviction requested before all observations were acknowledged')
                result = policy.evict(context)
                if not isinstance(result, dict) or set(result) != {'block_id', 'generation'}:
                    raise TypeError('evict must return {block_id: int, generation: int}; no fallback')
                if any(type(result[k]) is not int for k in result):
                    raise TypeError('Victim block ID and generation must be integers')
        response({'result': {'victim': result, 'through_event_id': last_event_id}})
    except BaseException as error:
        response({'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
