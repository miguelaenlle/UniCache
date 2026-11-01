"""Spool client for free local tests; the evaluator owns isolation and limits."""
import json
from pathlib import Path
import time
import uuid


def request(message):
    root = Path(__file__).resolve().parent
    key = uuid.uuid4().hex
    pending = root/'requests'/(key+'.tmp')
    pending.write_text(json.dumps(message))
    pending.rename(pending.with_suffix('.json'))
    response = root/'feedback'/(key+'.json')
    try:
        deadline = time.monotonic()+90
        while not response.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Local test response timed out; is the evaluator running?')
            time.sleep(.2)
        result = json.loads(response.read_text())
        print(result['output'], end='' if result['output'].endswith('\n') else '\n')
        return 0 if result['ok'] else 1
    finally:
        # Local tests leave no per-call log/spool records.
        pending.with_suffix('.json').unlink(missing_ok=True)
        response.unlink(missing_ok=True)
