"""Reproduce one selected policy's tune/eval scores without any Claude/API call."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import agent_budget_sweep as sweep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--blocks',type=int,required=True)
    p.add_argument('--trace',choices=sweep.TRACES,required=True)
    p.add_argument('--output',type=Path,required=True,help='New replay folder; must not exist')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',VLLM_USE_V1='0',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    item=json.loads((a.study/'agent-runs'/str(a.blocks)/(a.trace+'.json')).read_text())
    run=Path(item['run']);record=sweep.validate(sweep.read_run(run))
    configs,expected,_=sweep.prepared(a.study.resolve(),a.trace)
    evaluator=sweep.runner.Evaluator(SimpleNamespace(gpu_blocks=a.blocks),out,configs,expected)
    scores={}
    for split in ['tune','eval']:
        score,_=evaluator.replay('LRU',split,out/split,candidate=run/'policy.py')
        if score['token_weighted_hit_ratio']!=record['table']['Agent'][split]:
            raise ValueError('Selected-policy replay differs from recorded '+split+' score')
        scores[split]=score
    result={'passed':True,'gpu_blocks':a.blocks,'trace':a.trace,'policy_sha256':sweep.runner.digest(run/'policy.py'),'scores':scores}
    sweep.write(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
