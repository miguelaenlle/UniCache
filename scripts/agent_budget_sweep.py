"""Frozen-v1 agent/cache-budget experiment with resumable private baseline caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_agent import runner
from scripts.summarize_policy_agent import read_run, validate, LABELS, ORDER

ROOT = runner.ROOT
BUDGETS = [16384, 32768, 50000, 98304]
TRACES = list(runner.TRACE_CONFIGS)


class ProgressOnly:
    """Keep host progress; never persist coding-model text, thinking or tool calls."""
    prefixes=('Results: ','Baseline ','Replay ','Cached baseline ','Submission ',
              'Starting ','Finished.','HEARTBEAT ','ERROR:','Local test: ')

    def __init__(self,stream):
        self.stream=stream;self.buffers={};self.lock=threading.RLock();self.run_path=None

    def write(self,value):
        with self.lock:
            key=threading.get_ident();text=self.buffers.get(key,'')+value
            lines=text.split('\n');self.buffers[key]=lines.pop()
            for line in lines:
                if self.run_path is None and line.startswith('Results: '):
                    self.run_path=Path(line.split('Results: ',1)[1].strip())
                if line.startswith(self.prefixes):self.stream.write(line+'\n')
        return len(value)

    def flush(self):self.stream.flush()


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


def verify_feedback_barrier(run):
    """A CLI exit while submit.py still waits is not completed optimization."""
    # artifacts() rewrites private submissions.jsonl during finalization AFTER
    # stop.json even on healthy runs; only delivery/history timestamps are valid.
    files=[run/'sandbox/history.jsonl',
           *list((run/'sandbox/feedback').glob('submission-*.json'))]
    latest=max(p.stat().st_mtime for p in files if p.exists())
    stopped=(run/'stop.json').stat().st_mtime
    if stopped<latest:
        raise ValueError('agent_exited_while_submission_pending: stop preceded final submission feedback')
    return {'passed':True,'stop_after_final_feedback_seconds':stopped-latest}


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


def run_one(out, blocks, trace, cap, coding_model, normal_auth=False):
    terminal=sys.stdout;progress=ProgressOnly(terminal);sys.stdout=progress
    # Each coding case owns its process group so the deadline can stop only its
    # own runner/Claude/worker descendants. No other experiment process is killed.
    if os.getpgrp()!=os.getpid():os.setsid()
    done=threading.Event();started=time.monotonic()
    def heartbeat():
        while not done.wait(30):
            print(f'HEARTBEAT {blocks}/{trace}: {(time.monotonic()-started)/60:.1f} minutes; see Submission progress above',flush=True)
    def deadline():
        if not done.wait(7200):
            print('ERROR: 2-hour coding-case deadline exceeded; terminating this case without a valid completed result',flush=True)
            os.killpg(os.getpgrp(),signal.SIGTERM)
    threading.Thread(target=heartbeat,daemon=True).start()
    threading.Thread(target=deadline,daemon=True).start()
    # Host-only cache adapter: candidate replays ALWAYS execute normally.
    original = runner.Evaluator.replay
    def replay(self, policy, split, folder, candidate=None, use_map=None):
        if candidate is None:
            score, parent = cached(out, blocks, trace, policy, split)
            print(f'Cached baseline {policy}/{split}', flush=True)
            return score, parent
        return original(self, policy, split, folder, candidate=candidate, use_map=use_map)
    runner.Evaluator.replay = replay
    command=runner.claude_command
    def synchronous_command(sandbox,model):
        from policy_agent.client_python import install_client_python
        install_client_python(sandbox)
        cmd,env=command(sandbox,model)
        cmd += ['--disallowedTools','Edit(./bin/**),Write(./bin/**),Edit(./bin),Write(./bin)']
        env.update(CLAUDE_CODE_DISABLE_BACKGROUND_TASKS='1',
                   BASH_DEFAULT_TIMEOUT_MS='1800000',BASH_MAX_TIMEOUT_MS='1800000')
        return cmd,env
    runner.claude_command=synchronous_command
    # Freeze splits once before any parallel replay. Never rewrite files in use.
    runner.prepare = lambda args:prepared(out,trace)
    sys.argv = ['policy-agent','run','--trace',trace,
                '--serving_model','llama-3.2-1b-instruct','--coding_model',coding_model,
                '--start-policy','LRU','--compare','all','--require-baselines','OPT',
                '--max-submissions',str(cap),'--gpu-blocks',str(blocks)]
    if normal_auth:sys.argv.append('--normal-cli-auth')
    try:runner.main()
    finally:
        done.set();sys.stdout.flush();sys.stdout=terminal
    if progress.run_path is not None:verify_feedback_barrier(progress.run_path)


def agent(out, blocks, trace, cap, coding_model, normal_auth=False, wait=False):
    deadline=time.monotonic()+8*3600
    while not all(cache_path(out,blocks,trace,p,s).exists()
                  for p in runner.ALL_POLICIES for s in ['tune','eval']):
        if not wait:raise RuntimeError(f'Baselines pending for {blocks}/{trace}')
        completion=out/'baselines-completion.json'
        if completion.exists() and not json.loads(completion.read_text())['complete']:
            raise RuntimeError('Baseline stage failed; inspect baselines-completion.json')
        if time.monotonic()>deadline:raise TimeoutError('Baseline wait exceeded 8 hours')
        time.sleep(10)
    record = out/'agent-runs'/str(blocks)/(trace+'.json')
    if record.exists():
        validate(read_run(Path(json.loads(record.read_text())['run'])))
        return
    log = out/'agent-logs'/str(blocks)/(trace+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--stage','run-one',
               '--output',str(out),'--blocks',str(blocks),'--traces',trace,
               '--max-submissions',str(cap),'--coding-model',coding_model]
    if normal_auth:command.append('--normal-cli-auth')
    print(f'AGENT START {blocks}/{trace}; progress: {log}', flush=True)
    with log.open('w') as f:
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                   stdout=f, stderr=subprocess.STDOUT)
    runs = [line.split('Results: ',1)[1].strip() for line in log.read_text().splitlines() if line.startswith('Results: ')]
    if completed.returncode or len(runs) != 1:
        raise RuntimeError(f'Agent failed ({completed.returncode}): {log}')
    validated = validate(read_run(Path(runs[0])))
    barrier=verify_feedback_barrier(Path(runs[0]))
    write(record, {'run':runs[0], 'selected':validated['selected'],
                   'attempts':validated['attempts'], 'command':command,'feedback_barrier':barrier})
    print(f'AGENT DONE {blocks}/{trace}: {validated["attempts"]}/{cap} attempts', flush=True)


def report(out, budgets, traces):
    rows=[];head=[];origins=[]
    for blocks in budgets:
        for trace in traces:
            agent_record = out/'agent-runs'/str(blocks)/(trace+'.json')
            item=json.loads(agent_record.read_text()) if agent_record.exists() else None
            parsed = validate(read_run(Path(item['run']))) if item else None
            if item:origins.append({'gpu_blocks':blocks,'trace':trace,'run':item['run'],
                                    'reused_historical_coding':item.get('reused_historical_coding',False)})
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
                    online=[p for p in runner.ALL_POLICIES if p not in ['OPT','NONE']]
                    tune={p:cached(out,blocks,trace,p,'tune')[0]['token_weighted_hit_ratio']
                          for p in online if cache_path(out,blocks,trace,p,'tune').exists()}
                    best=max(online,key=lambda p:tune[p]) if len(tune)==len(online) else None
                    best_score=values.get(best) if best else None
                    simple=['LRU','LFU','FIFO']
                    best_simple=max(simple,key=lambda p:tune[p]) if all(p in tune for p in simple) else None
                    simple_score=values.get(best_simple) if best_simple else None
                    note=('agent exceeds measured offline reference' if parsed and values['Agent']>values['OPT']+1e-12
                          else 'nonpositive reference gap' if gap<=0 else '')
                    head.append([blocks,trace,split,values['LRU'],values['OPT'],values['Agent'],
                                 100*gap,100*gain if gain is not None else None,
                                 gain/gap if gain is not None and gap>0 else None,
                                 parsed['attempts'] if parsed else None,best,best_score,
                                 100*(best_score-values['LRU']) if best_score is not None else None,
                                 100*(values['Agent']-best_score) if parsed and best_score is not None else None,note,
                                 best_simple,simple_score,
                                 100*(values['Agent']-simple_score) if parsed and simple_score is not None else None,
                                 100*(values['OPT']-values['Agent']) if parsed else None])
    for name,columns,data in [
        ('scores.csv',['gpu_blocks','trace','split','policy','hit_ratio'],rows),
        ('headroom.csv',['gpu_blocks','trace','split','LRU','OPT','Agent','OPT_minus_LRU_pp',
                         'Agent_minus_LRU_pp','fraction_reference_gain_achieved','attempts',
                         'best_online_by_tune','best_online_hit_ratio','best_online_minus_LRU_pp',
                         'Agent_minus_best_online_pp','reference_gap_note',
                         'best_simple_by_tune','best_simple_hit_ratio','Agent_minus_best_simple_pp',
                         'OPT_minus_Agent_pp'],head)]:
        with (out/name).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(columns);w.writerows(data)
    md=['# V1 agent cache-budget experiment','',
        'Llama-3.2-1B-Instruct; 8-token blocks; LRU start; Claude Opus; 30-attempt cap (early stopping allowed).',
        'Policy selected by tune only; independent cold tune/eval caches. Blank cells mean unavailable (pending or failed), never zero.',
        'OPT is privileged offline reference headroom, not a guaranteed online-achievable improvement. All comparisons remain hidden from the coding agent.',
        'Best simple heuristic is selected from LRU, LFU and FIFO. Best heuristic is selected from all implemented online baselines (including UniCache); offline OPT and NONE are excluded.',
        'Both baseline identities maximize tune hit ratio separately for each workload and budget; their eval scores use those same identities. Ties use the first policy in the listed candidate order.',
        'Simple tie order: LRU, LFU, FIFO. Online tie order: '+', '.join(p for p in runner.ALL_POLICIES if p not in ['OPT','NONE'])+'.',
        'V1 retains starting-policy-dependent bookkeeping. V2 is a separate experiment.','']
    model=json.loads((ROOT/'models/Llama-3.2-1B-Instruct/config.json').read_text())
    head_dim=model.get('head_dim',model['hidden_size']//model['num_attention_heads'])
    block_bytes=2*model['num_hidden_layers']*model['num_key_value_heads']*head_dim*2*8
    capacity={'block_tokens':8,'kv_dtype':'float16','bytes_per_block':block_bytes,
              'budgets':{str(b):{'blocks':b,'modeled_kv_gib':b*block_bytes/2**30} for b in budgets}}
    write(out/'capacity.json',capacity)
    md += ['| GPU blocks | Modeled KV capacity |','|---:|---:|',
           *[f'| {b:,} | {b*block_bytes/2**30:.3f} GiB |' for b in budgets],
           '',f'Each 8-token FP16 block represents {block_bytes:,} bytes. These are modeled KV capacities, not measured Mac RAM or physical GPU allocations.',
           'The budgets are the preceding pressure-inducing pilot choices; oversized paper-normalized budgets were mostly saturated.','']
    reused=sum(x['reused_historical_coding'] for x in origins)
    md += [f'Completed coding runs: {len(origins)-reused} fresh, {reused} reused historical runs with freshly verified selected-policy tune/eval replays.',
           'Historical runs retain their original attempts, tune-based selection and source snapshots; they are not counted as new coding runs.',
           'The four explicit budgets are the previous pilot budgets, not paper-normalized A100 fractions.','']
    write(out/'agent-origins.json',origins)
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
    md += ['## Eval headroom','', '| Blocks | Trace | OPT − LRU | Agent − LRU | Best simple heuristic (tune) | Agent − best simple | Best heuristic (tune) | Agent − best heuristic | Fraction of reference gain |', '|---:|---|---:|---:|---|---:|---|---:|---:|']
    for row in head:
        blocks,trace,split,lru,opt,value,gap,gain,fraction,attempts=row[:10]
        if split=='eval':
            gained=f'{gain:.3f} pp' if gain is not None else '—'
            part=f'{fraction:.1%}' if fraction is not None else ('n/a' if gain is not None else '—')
            simple_gain=f'{row[17]:.3f} pp' if row[17] is not None else '—'
            heuristic_gain=f'{row[13]:.3f} pp' if row[13] is not None else '—'
            md.append(f'| {blocks:,} | {LABELS[trace]} | {gap:.3f} pp | {gained} | {row[15] or "—"} | {simple_gain} | {row[10] or "—"} | {heuristic_gain} | {part} |')
    md += ['', '## Agent versus offline OPT (eval)', '',
           'Change = Agent − LRU. Remaining headroom = OPT − Agent. Headroom attained = (Agent − LRU) / (OPT − LRU). '
           'The percentage is undefined when OPT equals LRU; missing agent results stay blank.', '',
           '| Modeled KV capacity (GiB) | Trace | LRU tune (%) | Agent tune (%) | Change tune (pp) | OPT − LRU tune (pp) | OPT − Agent tune (pp) | LRU eval (%) | Agent eval (%) | Change eval (pp) | OPT − LRU eval (pp) | OPT − Agent eval (pp remaining) | Eval headroom attained (%) |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    opt_rows=[]
    for row in head:
        blocks,trace,split,lru,opt,value,gap,gain,fraction,attempts=row[:10]
        if split!='eval':continue
        remaining=row[18]
        gib=blocks*block_bytes/2**30
        tune_values={r[3]:r[4] for r in rows if r[:3]==[blocks,trace,'tune']}
        tune_lru=tune_values['LRU'];tune_agent=tune_values.get('Agent')
        tune_gain=100*(tune_agent-tune_lru) if tune_agent is not None else None
        tune_gap=100*(tune_values['OPT']-tune_lru)
        tune_remaining=100*(tune_values['OPT']-tune_agent) if tune_agent is not None else None
        tune_remaining_text=f'{tune_remaining:.3f}' if tune_remaining is not None else '—'
        tune_agent_text=f'{100*tune_agent:.2f}%' if tune_agent is not None else '—'
        tune_gain_text=f'{tune_gain:+.3f}' if tune_gain is not None else '—'
        agent=f'{100*value:.2f}%' if value is not None else '—'
        change=f'{gain:+.3f}' if gain is not None else '—'
        left=f'{remaining:.3f}' if remaining is not None else '—'
        attained=f'{100*fraction:.1f}%' if fraction is not None else ('n/a' if value is not None else '—')
        md.append(f'| {gib:g} | {LABELS[trace]} | {100*tune_lru:.2f}% | {tune_agent_text} | {tune_gain_text} | {tune_gap:.3f} | {tune_remaining_text} | {100*lru:.2f}% | {agent} | {change} | {gap:.3f} | {left} | {attained} |')
        opt_rows.append([gib,trace,tune_lru,tune_agent,tune_gain,tune_gap,tune_remaining,
                         lru,value,gain,gap,remaining,100*fraction if fraction is not None else None])
    with (out/'agent-vs-opt-eval.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['modeled_kv_gib','trace','lru_tune_hit_ratio','agent_tune_hit_ratio','Agent_minus_LRU_tune_pp',
                    'OPT_minus_LRU_tune_pp','OPT_minus_Agent_tune_pp',
                    'lru_hit_ratio','agent_hit_ratio','Agent_minus_LRU_pp',
                    'OPT_minus_LRU_pp','OPT_minus_Agent_pp_remaining','headroom_attained_percent'])
        w.writerows(opt_rows)
    heuristic_md=['## Agent versus tune-selected best online heuristic', '',
                  'Capacity is modeled GiB; hit ratios are percentages; differences are percentage points (pp). '
                  'Best is selected on tune from all online baselines, excluding offline OPT and NONE. '
                  'The same selected heuristic is used on eval. Positive Best − Agent means the heuristic is ahead.', '',
                  '| KV capacity (GiB) | Trace | LRU tune (%) | Agent tune (%) | Change tune (pp) | OPT − LRU tune (pp) | OPT − Agent tune (pp) | LRU eval (%) | Agent eval (%) | Change eval (pp) | Best heuristic (selected on tune) | Best − Agent tune (pp) | Best − Agent eval (pp) |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    heuristic_rows=[]
    for row in head:
        blocks,trace,split,lru,opt,value,gap,gain,fraction,attempts=row[:10]
        if split!='eval':continue
        tune_row=next(r for r in head if r[:3]==[blocks,trace,'tune'])
        best=row[10]
        assert best==tune_row[10]
        tune_lru=tune_row[3];tune_agent=tune_row[5]
        tune_gain=tune_row[7]
        tune_gap=tune_row[6];tune_remaining=tune_row[18]
        tune_best=tune_row[11];eval_best=row[11]
        tune_difference=100*(tune_best-tune_agent) if tune_agent is not None else None
        eval_difference=100*(eval_best-value) if value is not None else None
        gib=blocks*block_bytes/2**30
        pct=lambda v:f'{100*v:.2f}' if v is not None else '—'
        delta=lambda v:f'{v:+.3f}' if v is not None else '—'
        heuristic_md.append(f'| {gib:g} | {LABELS[trace]} | {pct(tune_lru)} | {pct(tune_agent)} | {delta(tune_gain)} | {tune_gap:.3f} | {delta(tune_remaining)} | {pct(lru)} | {pct(value)} | {delta(gain)} | {best} | {delta(tune_difference)} | {delta(eval_difference)} |')
        heuristic_rows.append([gib,trace,tune_lru,tune_agent,tune_gain,tune_gap,tune_remaining,lru,value,gain,
                               best,tune_best,eval_best,tune_difference,eval_difference])
    with (out/'agent-vs-best-heuristic.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['modeled_kv_gib','trace','lru_tune_hit_ratio','agent_tune_hit_ratio','Agent_minus_LRU_tune_pp',
                    'OPT_minus_LRU_tune_pp','OPT_minus_Agent_tune_pp',
                    'lru_eval_hit_ratio','agent_eval_hit_ratio','Agent_minus_LRU_eval_pp',
                    'best_heuristic_selected_on_tune','best_heuristic_tune_hit_ratio','best_heuristic_eval_hit_ratio',
                    'Best_minus_Agent_tune_pp','Best_minus_Agent_eval_pp'])
        w.writerows(heuristic_rows)
    (out/'agent-vs-best-heuristic.md').write_text('\n'.join(heuristic_md)+'\n')
    md += ['',*heuristic_md]
    md += ['', '![Eval hit ratios across cache budgets](hit-ratio-by-budget.png)',
           '', 'Coincident curves indicate equal scores; open/different markers distinguish overlapping policies. Best baseline is selected on tune, then plotted on eval.',
           'Frozen policy replays are reproducible; a fresh Opus coding run may choose different policies or stop at a different attempt. See environment.json for the requested alias and CLI version.',
           'Any excluded bootstrap launches are explicitly recorded in aborted-coding-runs.json.', '']
    (out/'report.md').write_text('\n'.join(md)+'\n')
    if head:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(traces),figsize=(3.4*len(traces),4.8),squeeze=False)
        for ax,trace in zip(axes[0],traces):
            selected=sorted([r for r in head if r[1]==trace and r[2]=='eval'])
            for idx,label,color,marker,size,style in [
                (3,'LRU','#4c78a8','o',8,'-'),
                (11,'Best baseline (tune)','#54a24b','x',4,':'),
                (5,'Agent','#e45756','+',7,'--'),
                (4,'Offline OPT','#888888','*',7,'-.')]:
                xs=[r[0] for r in selected if r[idx] is not None];ys=[100*r[idx] for r in selected if r[idx] is not None]
                ax.plot(xs,ys,label=label,color=color,marker=marker,markersize=size,
                        markerfacecolor='none' if marker=='o' else color,linestyle=style,linewidth=1.4)
            uc=sorted((r[0],100*r[4]) for r in rows if r[1:4]==[trace,'eval','UniCache'] and r[4] is not None)
            ax.plot([x for x,y in uc],[y for x,y in uc],label='UniCache',color='#b279a2',
                    marker='D',markerfacecolor='none',markersize=6,linestyle='--',linewidth=1)
            ax.set_title(LABELS[trace]);ax.set_xlabel('GPU cache blocks');ax.grid(alpha=.3)
            ax.set_xticks(budgets,[f'{b:,}' for b in budgets],rotation=40,ha='right')
        axes[0][0].set_ylabel('Eval token-weighted hit ratio (%)')
        handles,labels=axes[0][-1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.03),ncol=5,fontsize=9)
        fig.text(.5,.005,'Overlapping curves mean equal measured scores; best baseline is selected on tune.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(out/'hit-ratio-by-budget.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','baselines','agents','report','run-one'],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blocks',nargs='+',type=int,default=BUDGETS)
    p.add_argument('--traces',nargs='+',choices=TRACES,default=TRACES)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--max-submissions',type=int,default=30)
    p.add_argument('--coding-model',default='claude:opus')
    p.add_argument('--normal-cli-auth',action='store_true')
    p.add_argument('--wait-for-baselines',action='store_true',help='Agent stage waits for each case while baseline stage runs separately')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.jobs<1 or min(a.blocks)<1:p.error('Positive jobs and budgets required')
    os.environ.update(PYTHONHASHSEED='0',VLLM_USE_V1='0',VLLM_NO_USAGE_STATS='1',
                      HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    if a.stage=='run-one':return run_one(out,a.blocks[0],a.traces[0],a.max_submissions,a.coding_model,a.normal_cli_auth)
    identity={'blocks':a.blocks,'traces':a.traces,'source_digest':source_digest(),
              'coding_model':a.coding_model,'max_submissions':a.max_submissions,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'interface':'v1','baseline_policies':runner.ALL_POLICIES}
    manifest=out/'manifest.json'
    if manifest.exists():
        stored=json.loads(manifest.read_text())
        if {k:stored[k] for k in identity}!=identity:raise ValueError('Experiment identity changed; use a new output folder')
    else:
        for trace in a.traces:
            configs,expected,m=runner.prepare(SimpleNamespace(trace=trace,sessions=None))
            write(out/'prepared'/(trace+'.json'),{'configs':{k:str(v) for k,v in configs.items()},
                                               'expected':expected,'manifest':m})
        write(manifest,identity)
    cases=[(b,t) for b in a.blocks for t in a.traces]
    for stage in (['baselines','agents'] if a.stage=='all' else [a.stage]):
        if stage=='report':continue
        if stage=='agents' and not a.normal_cli_auth and not any(os.environ.get(k) for k in ['ANTHROPIC_API_KEY','CLAUDE_CODE_OAUTH_TOKEN']):
            raise RuntimeError('Agent stage requires normal exported ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN; no credentials are read from other sources')
        fn=(lambda b,t:baselines(out,b,t)) if stage=='baselines' else (lambda b,t:agent(out,b,t,a.max_submissions,a.coding_model,a.normal_cli_auth,a.wait_for_baselines))
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
