"""Read-only sandbox client: evaluator owns numbering, budget, and scores."""
import argparse
import json
from pathlib import Path
import time
import uuid

p = argparse.ArgumentParser()
p.add_argument('--rationale-file')
p.add_argument('--done', action='store_true')
p.add_argument('--reason')
a = p.parse_args()
root = Path(__file__).resolve().parent
if a.done:
    if not a.reason:
        p.error('--done requires --reason')
    request = {'done': True, 'reason': a.reason}
else:
    if not a.rationale_file:
        p.error('--rationale-file is required')
    rationale = Path(a.rationale_file).read_text().strip()
    if not rationale:
        p.error('rationale must not be empty')
    request = {'source': (root / 'policy.py').read_text(), 'rationale': rationale}
key = uuid.uuid4().hex
pending = root / 'requests' / (key + '.tmp')
pending.write_text(json.dumps(request))
pending.rename(pending.with_suffix('.json'))
response = root / 'feedback' / (key + '.json')
while not response.exists():
    time.sleep(.5)
result = json.loads(response.read_text())
print(result['output'])
raise SystemExit(0 if result.get('ok') else 1)
