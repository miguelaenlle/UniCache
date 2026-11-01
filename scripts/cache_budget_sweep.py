"""1B cache-budget experiment on frozen tune/eval splits; no agent/API calls."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy_agent.runner import ROOT, TRACE_CONFIGS, Evaluator, prepare, digest
from scripts.hit_ratio_matrix import run_matrix

POLICIES=['LRU','LFU','FIFO','LeCaR','ARC','WA','SimpleHYBRID','OPT']
ONLINE=[p for p in POLICIES if p!='OPT']


def geometry(config,gpu_gib):
    h=config['hidden_size'];d=config['head_dim'];kv=config['num_key_value_heads']*d
    parameters=config['vocab_size']*h+config['num_hidden_layers']*(2*h*h+2*h*kv+3*h*config['intermediate_size']+2*h)+h
    if not config['tie_word_embeddings']:parameters+=config['vocab_size']*h
    block_bytes=2*config['num_hidden_layers']*config['num_key_value_heads']*d*2*8
    available=int(gpu_gib*2**30)-parameters*2
    if available<=0:raise ValueError('GPU memory must exceed weight storage')
    return {'parameters':parameters,'weight_bytes':parameters*2,'kv_block_bytes':block_bytes,'available_kv_bytes':available}


def tables(out,results,cases):
    (out/'results.json').write_text(json.dumps(results,indent=2))
    with (out/'hit_ratios.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['budget','gpu_blocks','trace','split','policy','hit_ratio','hit_tokens','prompt_tokens','requests','error'])
        for key,c in cases.items():
            for policy in POLICIES:
                r=results.get(key+'/'+policy,{})
                w.writerow([c['budget'],c['blocks'],c['trace'],c['split'],policy,r.get('hit_ratio',''),r.get('hit_tokens',''),r.get('prompt_tokens',''),r.get('requests',''),r.get('error','')])
    head=[]
    for key,c in cases.items():
        tune_key=key.rsplit('/',1)[0]+'/tune'
        available=[p for p in ONLINE if 'hit_ratio' in results.get(tune_key+'/'+p,{})]
        if len(available)!=len(ONLINE):continue
        best=max(ONLINE,key=lambda p:results[tune_key+'/'+p]['hit_ratio'])
        lru=results.get(key+'/LRU',{}).get('hit_ratio')
        opt=results.get(key+'/OPT',{}).get('hit_ratio')
        score=results.get(key+'/'+best,{}).get('hit_ratio')
        if None in (lru,opt,score):continue
        head.append([c['budget'],c['blocks'],c['trace'],c['split'],lru,best,score,opt,100*(opt-lru),100*(score-lru),100*(opt-score)])
    columns=['budget','gpu_blocks','trace','split','LRU','best_online_by_tune','best_online_hit_ratio','OPT','OPT_minus_LRU_pp','online_minus_LRU_pp','OPT_minus_online_pp']
    with (out/'headroom.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(head)
    lines=['# 1B cache-budget headroom','', 'All ratios are token weighted. Best online policy is selected on tune, then reported on eval.',
           'A = modeled 80 GiB device memory minus FP16 weight bytes; no measured activation/workspace reserve. OPT is an offline reference, not guaranteed online-achievable headroom.','',
           '| Trace | Budget | Blocks | LRU eval | Online policy (tune-selected) | Online eval | OPT eval | OPT − LRU (pp) | Online − LRU (pp) |',
           '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for b,blocks,t,s,l,p,v,o,g,online_gap,_ in head:
        if s=='eval':lines.append(f'| {t} | {b:.1f}A | {blocks:,} | {l:.4%} | {p} | {v:.4%} | {o:.4%} | {g:.4f} | {online_gap:.4f} |')
    lines+=['','No changes to model timing, scheduling, block size, dataset selection or arrival seeds across budgets. Independent cold caches and policy state for every replay. Source subsets match the preceding agent batch.','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--gpu-gib',type=float,default=80)
    p.add_argument('--output',type=Path)
    p.add_argument('--traces',nargs='+',choices=TRACE_CONFIGS,default=list(TRACE_CONFIGS))
    p.add_argument('--sessions',type=int,help='Explicit pilot subset override; omitted for the full experiment')
    a=p.parse_args()
    if a.jobs<1:p.error('jobs must be positive')
    os.chdir(ROOT)
    if sys.version_info<(3,12):p.error('Python 3.12 required for OPT hashing')
    os.environ.update(PYTHONHASHSEED='0',VLLM_USE_V1='0',VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    config=ROOT/'models/Llama-3.2-1B-Instruct/config.json'
    sizes=geometry(json.loads(config.read_text()),a.gpu_gib)
    budgets={f:int(f*sizes['available_kv_bytes']//sizes['kv_block_bytes']) for f in [.5,.7,.9]}
    out=(a.output or ROOT/'result/cache-budget-sweep'/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d/%H%M%S')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    print('Results:',out,flush=True);print('GPU blocks:',budgets,flush=True)
    manifests={};inputs={}
    for trace in a.traces:
        cfg,expected,manifest=prepare(SimpleNamespace(trace=trace,sessions=a.sessions))
        inputs[trace]=(cfg,expected);manifests[trace]=manifest
    manifest={'geometry':sizes,'gpu_gib':a.gpu_gib,'kv_dtype':'float16','block_size':8,'tensor_parallel':1,'budgets':budgets,
              'traces':manifests,'policies':POLICIES,'args':vars(a)|{'output':str(out)},
              'model_config_sha256':digest(config),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'timing':{'ttft':'3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)','tpot':.01},
              'note':'Idealized memory conversion; no physical GPU profiling. Frozen tune/eval trace inputs; same Llama-3.2-1B timing function across budgets.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    cases={};evaluators={}
    for fraction,blocks in budgets.items():
        for trace in a.traces:
            cfg,expected=inputs[trace]
            e=Evaluator(SimpleNamespace(gpu_blocks=blocks),out,cfg,expected)
            for split in ['tune','eval']:
                key=f'{fraction:.1f}A/{trace}/{split}'
                cases[key]={'budget':fraction,'blocks':blocks,'trace':trace,'split':split};evaluators[key]=e
    results={};cancelled=threading.Event()
    def replay(key,policy,use_map):
        c=cases[key]
        score,parent=evaluators[key].replay(policy,c['split'],out/'replays'/key/policy,use_map=use_map)
        # run_matrix reads native summaries for regular policies; NONE only releases OPT.
        return parent/'summary.json'
    def record(trace,policy,value):
        results[trace+'/'+policy]=value;tables(out,results,cases)
        print(trace,policy, f"{value['hit_ratio']:.4%}" if 'hit_ratio' in value else value['error'],flush=True)
    tables(out,results,cases)
    try:run_matrix(cases,POLICIES,a.jobs,replay,record,cancelled)
    except KeyboardInterrupt:
        cancelled.set();raise
    failures=[k for k,v in results.items() if 'error' in v]
    done={'passed':not failures and len(results)==len(cases)*len(POLICIES),'completed':len(results),'expected':len(cases)*len(POLICIES),'failures':failures}
    (out/'completion.json').write_text(json.dumps(done,indent=2))
    print(done,flush=True);print((out/'report.md').read_text(),flush=True)
    raise SystemExit(0 if done['passed'] else 1)

if __name__=='__main__':main()
