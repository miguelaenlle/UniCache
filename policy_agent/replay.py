"""Trusted replay entrypoint with independent request completion accounting."""
import json
import os
from pathlib import Path
import runpy
import sys

# Validate startup-only mode in the same isolated runtime before constructing an
# engine. NONE is a real disabled-cache control, not repeated idle-block eviction.
if os.environ.get('UNICACHE_POLICY_INTERFACE') == 'v2':
    from policy_agent.v2_bridge import V2Remote
    def cli_value(name, default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    source = Path(os.environ['UNICACHE_POLICY_CANDIDATE']).read_text()
    probe = V2Remote(source, {'api_version':2,
                            'capacity':int(cli_value('--simulator-num-gpu-blocks', 50000)),
                            'block_size':int(cli_value('--block-size', 8)), 'cache_id':'gpu:0'})
    try:
        mode = probe.initial['cache_mode']
    finally:
        probe.close()
    if mode == 'disabled' and '--enable-prefix-caching' in sys.argv:
        sys.argv.remove('--enable-prefix-caching')

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
