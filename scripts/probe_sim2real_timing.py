"""Diagnose timing sensitivity; these runs are not A100-calibrated predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--tpots', type=float, nargs='+', default=[.001, .003])
    args = p.parse_args()
    study = args.study.resolve()
    root = Path(__file__).resolve().parents[1]
    source = study/'replays/demo-blocks512/LRU'
    base = json.loads((source/'command.json').read_text())
    target = study/'diagnostics/timing-sensitivity'
    target.mkdir(parents=True, exist_ok=False)
    def run(tpot):
        folder = target/f'tpot-{tpot:g}'
        folder.mkdir()
        command = list(base)
        command[command.index('--simulator-tpot')+1] = str(tpot)
        command[command.index('--simulator-output-dir')+1] = str(folder)
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        expected = study/'inputs/demo-blocks512/expected-tokens.json'
        env = dict(os.environ, PYTHONHASHSEED='0', VLLM_USE_V1='0', HF_HUB_OFFLINE='1',
                   VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                   COLLECT_TRACE='0', UNICACHE_EXPECTED_TOKENS=str(expected),
                   UNICACHE_COMPLETIONS=str(folder/'completed.json'))
        for key in ['UNICACHE_POLICY_CANDIDATE','UNICACHE_POLICY_INTERFACE','POLICY_AGENT_INTERFACE']:
            env.pop(key,None)
        for key in ['MPLCONFIGDIR','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT']:
            path=folder/key;path.mkdir();env[key]=str(path)
        with (folder/'run.log').open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        paths=list(folder.glob('*/summary.json'))
        if len(paths)!=1:raise ValueError('Expected one summary')
        summary=json.loads(paths[0].read_text())
        completed=json.loads((folder/'completed.json').read_text())
        if len(completed)!=82:raise ValueError('Incomplete ShareGPT replay')
        record={'tpot_seconds':tpot,'hit_ratio_percent':100*summary['hit_rate_summary']['global_hit_rate'],
                'simulated_duration_seconds':summary['global_time'],'requests':len(completed),
                'purpose':'Sensitivity only; not calibration or replacement of primary results'}
        (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(record,flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,args.tpots))
    (target/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
