"""Native replay with exact token and completion checks against a frozen subset."""
import hashlib
import json
import os
from pathlib import Path
import runpy

from vllm.simulator.simulator import Simulator

expected = json.loads(Path(os.environ['UNICACHE_EXPECTED_TOKENS']).read_text())
completed = {}
original = Simulator._maybe_schedule_next_turn


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def observe(self, output):
    if output.finished:
        key = self.output_simulator.request_id_2_conv_key_map[output.request_id]
        if key not in expected or key in completed:
            raise RuntimeError('Unexpected or duplicate completed request: ' + key)
        prompt = list(output.prompt_token_ids)
        response = list(output.outputs[0].token_ids)
        reference = expected[key]
        if prompt != reference['input_ids'] or response != reference['response_ids']:
            raise RuntimeError('Prompt/output tokens differ from frozen trace: ' + key)
        completed[key] = {'prompt_tokens': len(prompt), 'output_tokens': len(response),
                          'prompt_sha256': token_hash(prompt), 'output_sha256': token_hash(response)}
    return original(self, output)


Simulator._maybe_schedule_next_turn = observe
runpy.run_path('run_simulation.py', run_name='__main__')
if set(completed) != set(expected):
    raise RuntimeError('Replay omitted expected requests')
Path(os.environ['UNICACHE_COMPLETIONS']).write_text(json.dumps(completed, indent=2) + '\n')
