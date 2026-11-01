"""Small real replay preflight: all prepared traces and offline OPT. No Claude calls."""
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_agent.runner import Evaluator, TRACE_CONFIGS, prepare

root=Path('result/policy-agent-preflight')/datetime.now().strftime('%Y%m%d-%H%M%S')
root=root.resolve();root.mkdir(parents=True)

def check(trace):
    args=Namespace(trace=trace,sessions=2,gpu_blocks=50000,start_policy='LRU',max_submissions=30)
    configs,expected,manifest=prepare(args)
    out=root/trace;out.mkdir()
    e=Evaluator(args,out,configs,expected)
    profile,parent=e.replay('NONE','tune',out/'profile')
    score,_=e.replay('OPT','tune',out/'opt',use_map=parent/'use_map.json')
    return {'trace':trace,'passed':True,'requests':score['requests'],'none_hit_ratio':profile['token_weighted_hit_ratio'],'opt_hit_ratio':score['token_weighted_hit_ratio']}

results={}
with ThreadPoolExecutor(max_workers=3) as pool:
    futures={pool.submit(check,t):t for t in TRACE_CONFIGS}
    for f in as_completed(futures):
        trace=futures[f]
        try: results[trace]=f.result()
        except Exception as error: results[trace]={'passed':False,'error':str(error)}
        (root/'checks.json').write_text(json.dumps(results,indent=2))
        print(trace,results[trace],flush=True)
print('Preflight:',root,flush=True)
raise SystemExit(0 if all(r['passed'] for r in results.values()) else 1)
