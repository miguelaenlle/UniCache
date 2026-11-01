"""Verify and freshly replay historical 50k agent winners before reusing coding runs."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts import agent_budget_sweep as sweep


def verify_code(commit):
    forbidden=['vllm','run_simulation.py','policy_agent/bridge.py','policy_agent/worker.py',
               'policy_agent/replay.py','policy_agent/starter.py','policy_agent/smoke_suite.py']
    if subprocess.check_output(['git','diff',commit,'HEAD','--',*forbidden],cwd=sweep.ROOT):
        raise ValueError('Historical simulator/interface code differs')
    previous=subprocess.check_output(['git','show',commit+':policy_agent/runner.py'],cwd=sweep.ROOT,text=True)
    current=(sweep.ROOT/'policy_agent/runner.py').read_text()
    for name in ['prepare','Evaluator','PROMPT','API']:
        def selected(src):
            for n in ast.parse(src).body:
                if getattr(n,'name',None)==name:return n
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return n
            raise ValueError('Missing '+name)
        if ast.dump(selected(previous))!=ast.dump(selected(current)):
            raise ValueError('Historical evaluator/prompt/interface differs: '+name)


def import_run(study,source):
    run=Path(source);data=sweep.read_run(run);record=sweep.validate(data)
    meta=json.loads(data['run.json']);args=meta['args'];trace=args['trace'];blocks=args['gpu_blocks']
    if (blocks,args['start_policy'],args['coding_model'],args['max_submissions'])!=(50000,'LRU','claude:opus',30):
        raise ValueError('Historical settings differ')
    if meta['python']!=sys.version:raise ValueError('Python version differs')
    verify_code(meta['git_commit'])
    configs,expected,new=sweep.prepared(study,trace)
    old=meta['trace']
    for key in ['source_session_limit','chronology','split']:
        if old[key]!=new[key]:raise ValueError('Historical split definition differs: '+key)
    original_source=Path(old['source'])
    if sweep.runner.digest(original_source)!=old['source_sha256']:
        raise ValueError('Original historical source changed')
    for split in ['tune','eval']:
        before=old['configs'][split];after=new['configs'][split]
        if before['data_sha256']!=after['data_sha256']:raise ValueError('Split token data differs')
        original_config=Path(before['path'])
        if sweep.runner.digest(original_config)!=before['sha256']:raise ValueError('Historical config changed')
        first=yaml.safe_load(original_config.read_text());second=yaml.safe_load(Path(after['path']).read_text())
        for cfg in [first,second]:
            for task in cfg['tasks']:task['data_path']='<frozen split>'
        if first!=second:raise ValueError('Arrival/scheduler/task configuration differs')
    for relative,digest in meta['model_checksums'].items():
        if sweep.runner.digest(sweep.ROOT/relative)!=digest:raise ValueError('Model metadata differs')
    replay_scores={};evaluator=sweep.runner.Evaluator(SimpleNamespace(gpu_blocks=blocks),study,configs,expected)
    for split in ['tune','eval']:
        folder=study/'historical-verification'/trace/split
        cache=study/'historical-verification'/trace/(split+'.json')
        if cache.exists():
            score=json.loads(cache.read_text())
        else:
            score,_=evaluator.replay('LRU',split,folder,candidate=run/'policy.py')
            sweep.write(cache,score)
        if score['token_weighted_hit_ratio']!=record['table']['Agent'][split]:
            raise ValueError('Historical selected policy no longer reproduces exactly: '+trace+'/'+split)
        replay_scores[split]=score
    target=study/'agent-runs'/str(blocks)/(trace+'.json')
    if target.exists():raise ValueError('Refuse to overwrite an existing agent run record')
    snapshot=study/'historical-coding-runs'/trace
    for name,raw in data.items():
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    sweep.write(target,{'run':str(snapshot),'source_run':str(run),
                       'selected':record['selected'],'attempts':record['attempts'],
                       'reused_historical_coding':True,'historical_git_commit':meta['git_commit'],
                       'fresh_replay':replay_scores,
                       'checks':['source checksum','split token checksums','normalized configs and frozen arrivals',
                                 'simulator/controller source parity','evaluator/prepare/prompt/API AST parity',
                                 'Python version','model metadata checksums','tune-only selection/source digest',
                                 'fresh exact tune/eval score reproduction'],
                       'snapshot_sha256':{name:sweep.runner.digest(snapshot/name) for name in data}})
    print(f'Historical {trace}/50000: source and selected-policy replays verified exactly',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--summary-manifest',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();m=json.loads(a.summary_manifest.read_text())
    def run(item):return import_run(a.study.resolve(),item['source_run'])
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(run,m['runs']))


if __name__=='__main__':main()
