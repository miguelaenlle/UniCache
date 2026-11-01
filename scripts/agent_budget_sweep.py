"""Frozen-v1 agent/cache-budget experiment with resumable private baseline caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_agent import runner
from scripts.summarize_policy_agent import read_run, validate, LABELS, ORDER

ROOT = runner.ROOT
BUDGETS = [16384, 32768, 50000, 98304]
TRACES = list(runner.TRACE_CONFIGS)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write(path, value)


def source_digest():
    h = hashlib.sha256()
    for prefix in ['policy_agent', 'vllm']:
        for p in sorted((ROOT/prefix).rglob('*.py')):
            h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    h.update((ROOT/'run_simulation.py').read_bytes())
    return h.hexdigest()


def cache_path(out, blocks, trace, policy, split):
    return out/'baselines'/str(blocks)/trace/policy/(split+'.json')


def cached(out, blocks, trace, policy, split):
    p = cache_path(out, blocks, trace, policy, split)
    item = json.loads(p.read_text())
    parent = Path(item['parent'])
    for name, expected in item['sha256'].items():
        if runner.digest(parent/name) != expected:
            raise ValueError('Baseline artifact checksum mismatch: '+str(parent/name))
    return item['score'], parent


def prepared(out, trace):
    item=json.loads((out/'prepared'/(trace+'.json')).read_text())
    for split,info in item['manifest']['configs'].items():
        if runner.digest(info['path'])!=info['sha256']:
            raise ValueError('Prepared config checksum mismatch')
        data=Path(info['path']).with_suffix('.jsonl')
        if runner.digest(data)!=info['data_sha256']:
            raise ValueError('Prepared split checksum mismatch')
    return {k:Path(v) for k,v in item['configs'].items()},item['expected'],item['manifest']


def baselines(out, blocks, trace):
    configs, expected, manifest = prepared(out,trace)
    evaluator = runner.Evaluator(SimpleNamespace(gpu_blocks=blocks), out, configs, expected)
    for policy in ['NONE', *[p for p in runner.ALL_POLICIES if p != 'NONE']]:
        for split in ['tune', 'eval']:
            target = cache_path(out, blocks, trace, policy, split)
            if target.exists():
                cached(out, blocks, trace, policy, split)
                continue
            use_map = None
            if policy == 'OPT':
                _, profile = cached(out, blocks, trace, 'NONE', split)
                use_map = profile/'use_map.json'
            folder = out/'baseline-replays'/str(blocks)/trace/policy/split
            if folder.exists():
                raise ValueError('Incomplete replay exists; inspect and move it before retrying: '+str(folder))
            score, parent = evaluator.replay(policy, split, folder, use_map=use_map)
            files = ['summary.json', 'simulator_data.pkl']
            if policy == 'NONE':files += ['use_map.json']
            write(target, {'score':score, 'parent':str(parent),
                           'sha256':{name:runner.digest(parent/name) for name in files},
                           'trace':manifest})
            print(f'BASELINE {blocks}/{trace}/{policy}/{split}: {score["token_weighted_hit_ratio"]:.4%}', flush=True)


def run_one(out, blocks, trace, cap, coding_model):
    # Host-only cache adapter: candidate replays ALWAYS execute normally.
    original = runner.Evaluator.replay
    def replay(self, policy, split, folder, candidate=None, use_map=None):
        if candidate is None:
            score, parent = cached(out, blocks, trace, policy, split)
            print(f'Cached baseline {policy}/{split}', flush=True)
            return score, parent
        return original(self, policy, split, folder, candidate=candidate, use_map=use_map)
    runner.Evaluator.replay = replay
    # Freeze splits once before any parallel replay. Never rewrite files in use.
    runner.prepare = lambda args:prepared(out,trace)
    sys.argv = ['policy-agent','run','--trace',trace,
                '--serving_model','llama-3.2-1b-instruct','--coding_model',coding_model,
                '--start-policy','LRU','--compare','all','--require-baselines','OPT',
                '--max-submissions',str(cap),'--gpu-blocks',str(blocks)]
    runner.main()


def agent(out, blocks, trace, cap, coding_model):
    record = out/'agent-runs'/str(blocks)/(trace+'.json')
    if record.exists():
        validate(read_run(Path(json.loads(record.read_text())['run'])))
        return
    log = out/'agent-logs'/str(blocks)/(trace+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--stage','run-one',
               '--output',str(out),'--blocks',str(blocks),'--traces',trace,
               '--max-submissions',str(cap),'--coding-model',coding_model]
    print(f'AGENT START {blocks}/{trace}; progress: {log}', flush=True)
    with log.open('w') as f:
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                   stdout=f, stderr=subprocess.STDOUT)
    runs = [line.split('Results: ',1)[1].strip() for line in log.read_text().splitlines() if line.startswith('Results: ')]
    if completed.returncode or len(runs) != 1:
        raise RuntimeError(f'Agent failed ({completed.returncode}): {log}')
    validated = validate(read_run(Path(runs[0])))
    write(record, {'run':runs[0], 'selected':validated['selected'],
                   'attempts':validated['attempts'], 'command':command})
    print(f'AGENT DONE {blocks}/{trace}: {validated["attempts"]}/{cap} attempts', flush=True)


def report(out, budgets, traces):
    rows=[];head=[]
    for blocks in budgets:
        for trace in traces:
            agent_record = out/'agent-runs'/str(blocks)/(trace+'.json')
            parsed = validate(read_run(Path(json.loads(agent_record.read_text())['run']))) if agent_record.exists() else None
            for split in ['tune','eval']:
                values={}
                for policy in runner.ALL_POLICIES:
                    if cache_path(out,blocks,trace,policy,split).exists():
                        score,_ = cached(out,blocks,trace,policy,split)
                        values[policy]=score['token_weighted_hit_ratio']
                values['Agent']=parsed['table']['Agent'][split] if parsed else None
                for policy,value in values.items():rows.append([blocks,trace,split,policy,value])
                if {'LRU','OPT'}.issubset(values):
                    gap=values['OPT']-values['LRU'];gain=values['Agent']-values['LRU'] if parsed else None
                    head.append([blocks,trace,split,values['LRU'],values['OPT'],values['Agent'],
                                 100*gap,100*gain if gain is not None else None,
                                 gain/gap if gain is not None and gap>0 else None,
                                 parsed['attempts'] if parsed else None])
    for name,columns,data in [
        ('scores.csv',['gpu_blocks','trace','split','policy','hit_ratio'],rows),
        ('headroom.csv',['gpu_blocks','trace','split','LRU','OPT','Agent','OPT_minus_LRU_pp',
                         'Agent_minus_LRU_pp','fraction_reference_gain_achieved','attempts'],head)]:
        with (out/name).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(columns);w.writerows(data)
    md=['# V1 agent cache-budget experiment','',
        'Llama-3.2-1B-Instruct; 8-token blocks; LRU start; Claude Opus; 30-attempt cap (early stopping allowed).',
        'Policy selected by tune only; independent cold tune/eval caches. Blank cells mean pending, never zero.',
        'OPT is privileged offline reference headroom, not a guaranteed online-achievable improvement. All comparisons remain hidden from the coding agent.',
        'V1 retains starting-policy-dependent bookkeeping. V2 is a separate experiment.','']
    for blocks in budgets:
        for split in ['tune','eval']:
            md += [f'## {blocks:,} GPU blocks / {split}','',
                   '| Policy | '+' | '.join(LABELS[t] for t in traces)+' |',
                   '|---|'+'---:|'*len(traces)]
            for policy in ORDER:
                cells=[]
                for trace in traces:
                    match=[r[4] for r in rows if r[:4]==[blocks,trace,split,policy]]
                    cells.append(f'{100*match[0]:.4f}%' if match and match[0] is not None else '—')
                md.append('| '+policy+' | '+' | '.join(cells)+' |')
            md += ['']
    md += ['## Eval headroom','', '| Blocks | Trace | OPT − LRU | Agent − LRU | Fraction of reference gain |', '|---:|---|---:|---:|---:|']
    for blocks,trace,split,lru,opt,value,gap,gain,fraction,attempts in head:
        if split=='eval':md.append(f'| {blocks:,} | {LABELS[trace]} | {gap:.3f} pp | '+(f'{gain:.3f} pp | {fraction:.1%} |' if fraction is not None else '— | — |'))
    (out/'report.md').write_text('\n'.join(md)+'\n')
    if head:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(traces),figsize=(3.4*len(traces),4),squeeze=False)
        for ax,trace in zip(axes[0],traces):
            selected=sorted([r for r in head if r[1]==trace and r[2]=='eval'])
            for idx,label in [(3,'LRU'),(4,'Offline OPT'),(5,'Agent')]:
                xs=[r[0] for r in selected if r[idx] is not None];ys=[100*r[idx] for r in selected if r[idx] is not None]
                ax.plot(xs,ys,'o-',label=label)
            ax.set_title(LABELS[trace]);ax.set_xlabel('GPU cache blocks');ax.grid(alpha=.3);ax.tick_params(axis='x',rotation=35)
        axes[0][0].set_ylabel('Eval token-weighted hit ratio (%)');axes[0][-1].legend()
        fig.tight_layout();fig.savefig(out/'hit-ratio-by-budget.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','baselines','agents','report','run-one'],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blocks',nargs='+',type=int,default=BUDGETS)
    p.add_argument('--traces',nargs='+',choices=TRACES,default=TRACES)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--max-submissions',type=int,default=30)
    p.add_argument('--coding-model',default='claude:opus')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.jobs<1 or min(a.blocks)<1:p.error('Positive jobs and budgets required')
    os.environ.update(PYTHONHASHSEED='0',VLLM_USE_V1='0',VLLM_NO_USAGE_STATS='1',
                      HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    if a.stage=='run-one':return run_one(out,a.blocks[0],a.traces[0],a.max_submissions,a.coding_model)
    identity={'blocks':a.blocks,'traces':a.traces,'source_digest':source_digest(),
              'coding_model':a.coding_model,'max_submissions':a.max_submissions,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'interface':'v1','baseline_policies':runner.ALL_POLICIES}
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise ValueError('Experiment identity changed; use a new output folder')
    else:
        for trace in a.traces:
            configs,expected,m=runner.prepare(SimpleNamespace(trace=trace,sessions=None))
            write(out/'prepared'/(trace+'.json'),{'configs':{k:str(v) for k,v in configs.items()},
                                               'expected':expected,'manifest':m})
        write(manifest,identity)
    cases=[(b,t) for b in a.blocks for t in a.traces]
    for stage in (['baselines','agents'] if a.stage=='all' else [a.stage]):
        if stage=='report':continue
        if stage=='agents' and not any(os.environ.get(k) for k in ['ANTHROPIC_API_KEY','CLAUDE_CODE_OAUTH_TOKEN']):
            raise RuntimeError('Agent stage requires normal exported ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN; no credentials are read from other sources')
        fn=(lambda b,t:baselines(out,b,t)) if stage=='baselines' else (lambda b,t:agent(out,b,t,a.max_submissions,a.coding_model))
        errors=[]
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures={pool.submit(fn,b,t):(b,t) for b,t in cases}
            for f in as_completed(futures):
                try:f.result()
                except Exception as e:errors.append({'case':futures[f],'error':str(e)});print(errors[-1],flush=True)
        report(out,a.blocks,a.traces)
        write(out/(stage+'-completion.json'),{'complete':not errors,'errors':errors,'cases':len(cases)})
        if errors:raise SystemExit(1)
    report(out,a.blocks,a.traces)
    print('Report:',out/'report.md',flush=True)


if __name__=='__main__':main()
