"""Trusted replay entrypoint with independent request completion accounting."""
import json
import os
from pathlib import Path
import runpy
import sys
from vllm.simulator.simulator import Simulator

original = Simulator._maybe_schedule_next_turn
completed = {}
completed_prompts = {}
def observe(self, output):
    if output.finished:
        key = self.output_simulator.request_id_2_conv_key_map[output.request_id]
        if key in completed:
            raise RuntimeError('Duplicate completed request: ' + key)
        completed[key] = len(output.outputs[0].token_ids)
        completed_prompts[key] = len(output.prompt_token_ids)
    return original(self, output)
Simulator._maybe_schedule_next_turn = observe
runpy.run_path('run_simulation.py', run_name='__main__')
Path(os.environ['UNICACHE_COMPLETIONS']).write_text(json.dumps(completed))

Path(os.environ['UNICACHE_COMPLETIONS']).with_name('completed-prompts.json').write_text(json.dumps(completed_prompts))
