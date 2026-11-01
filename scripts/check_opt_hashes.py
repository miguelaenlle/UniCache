"""Check real allocator/profile hashes in separate simulator processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if sys.version_info < (3, 12):
        sys.exit('Use .venv-sim312/bin/python: OPT requires Python 3.12.')
    code = '''
import json
from vllm.core.block.prefix_caching_block import PrefixCachingBlock
from vllm.simulator.opt_utils import get_all_token_hash
tokens = list(range(32))
profile = get_all_token_hash(tokens, 8)
allocator = []
previous = None
for i in range(4):
    previous = PrefixCachingBlock.hash_block_tokens(i == 0, previous, tokens[i*8:(i+1)*8])
    allocator.append(previous)
assert allocator == profile, (allocator, profile)
print('HASH_RESULT=' + json.dumps(allocator))
'''
    root = Path(__file__).resolve().parents[1]
    values = []
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, PYTHONHASHSEED='0', VLLM_USE_V1='0', VLLM_NO_USAGE_STATS='1',
                   HF_HUB_OFFLINE='1', MPLCONFIGDIR=cache, VLLM_CONFIG_ROOT=cache,
                   VLLM_CACHE_ROOT=cache)
        for _ in range(3):
            output = subprocess.check_output([sys.executable, '-c', code], cwd=root,
                                             env=env, text=True)
            values.append(json.loads(next(line.removeprefix('HASH_RESULT=')
                                          for line in output.splitlines()
                                          if line.startswith('HASH_RESULT='))))
    assert all(value == values[0] for value in values), values
    print('PASS: allocator and profile hashes match in three separate Python processes.')


if __name__ == '__main__':
    main()
