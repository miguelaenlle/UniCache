"""Run a custom Python script through the evaluator's offline local-test sandbox."""
import argparse
from local_client import request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('script', help='Python script inside the workspace')
p.add_argument('args', nargs=argparse.REMAINDER)
a = p.parse_args()
raise SystemExit(request({'action':'local_test', 'script':a.script, 'args':a.args}))
