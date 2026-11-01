"""Pair native and standalone-v2 replays; compare every eviction and trusted scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_agent.v2_policies import POLICIES, source_for_policy
from policy_agent.runner import prepare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(policy, case, out, mode, source, blocks):
    folder = out / case['name'] / policy / mode
    folder.mkdir(parents=True)
    env = dict(os.environ, VLLM_USE_V1='0', VLLM_NO_USAGE_STATS='1', PYTHONHASHSEED='0',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
               COLLECT_TRACE='0', UNICACHE_COMPLETIONS=str(folder/'completed.json'),
               UNICACHE_POLICY_DECISIONS=str(folder/'decisions.jsonl'),
               UNICACHE_POLICY_SHADOW_FAILURE=str(folder/'first-divergence.json'))
    for key in ('UNICACHE_POLICY_CANDIDATE', 'UNICACHE_POLICY_INTERFACE', 'UNICACHE_POLICY_OBSERVE'):
        env.pop(key, None)
    if mode == 'v2':
        env.update(UNICACHE_POLICY_CANDIDATE=str(source), UNICACHE_POLICY_INTERFACE='v2')
    elif mode == 'shadow':
        env.update(UNICACHE_POLICY_CANDIDATE=str(source), UNICACHE_POLICY_OBSERVE='1')
    for key in ('MPLCONFIGDIR','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT'):
        path = folder / key
        path.mkdir()
        env[key] = str(path)
    native = {'UniCache':'SimpleHYBRID'}.get(policy, policy)
    if mode == 'v2':
        native = case.get('v2_host_policy','LRU')
    cmd = [sys.executable,'-m','policy_agent.replay','--model',str(ROOT/'models/Llama-3.2-1B-Instruct'),
           '--device','cpu','--max-num-seqs',str(case.get('max_num_seqs',1)),'--max-model-len','131072','--block-size','8',
           '--simulator-enable-virtual-time','--simulator-num-gpu-blocks',str(blocks),
           '--simulator-num-cpu-blocks','50000','--eviction-policy',native,
           '--simulator-ttft','3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)',
           '--simulator-tpot','.01','--simulator-output-dir',str(folder),
           '--simulator-task-config',str(case['config']),'--log-level','WARNING']
    if policy != 'NONE':
        cmd += ['--enable-prefix-caching']
    (folder/'command.json').write_text(json.dumps(cmd,indent=2))
    with (folder/'run.log').open('w') as log:
        result = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:
        raise RuntimeError(f'{folder}: replay exited {result.returncode}: '+(folder/'run.log').read_text()[-2500:])
    summaries = list(folder.glob('**/summary.json'))
    assert len(summaries)==1, folder
    summary = json.loads(summaries[0].read_text())
    completed = json.loads((folder/'completed.json').read_text())
    prompts = json.loads((folder/'completed-prompts.json').read_text())
    expected = case['expected']
    assert completed == {k:v[1] for k,v in expected.items()}, 'completion ledger mismatch'
    assert prompts == {k:v[0] for k,v in expected.items()}, 'prompt ledger mismatch'
    counts = summary.get('hit_rate_summary', {})
    decisions_path=folder/'decisions.jsonl'
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()] if decisions_path.exists() else []
    return {'hit_tokens':counts.get('global_hit_tokens_num',0),'prompt_tokens':counts.get('global_tokens',sum(prompts.values())),
            'requests':len(completed),'decisions':decisions,'completed':completed,'prompts':prompts}


def compare(policy, case, out, blocks):
    source = out/'sources'/f'{policy}.py'
    try:
        native = replay(policy,case,out,'native',source,blocks)
        candidate = replay(policy,case,out,'v2',source,blocks)
        shadow = replay(policy,case,out,'shadow',source,blocks) if case.get('shadow') and policy != 'NONE' else None
        nd, cd = native['decisions'],candidate['decisions']
        first = None
        for i in range(max(len(nd),len(cd))):
            a,b = (nd[i] if i<len(nd) else None),(cd[i] if i<len(cd) else None)
            if a != b:
                first = {'index':i,'native':a,'v2':b}
                break
        scorekeys=('hit_tokens','prompt_tokens','requests','completed','prompts')
        scores_match=all(native[k]==candidate[k] for k in scorekeys)
        shadow_match = shadow is None or (shadow['decisions']==nd and all(shadow[k]==native[k] for k in scorekeys))
        status='ok' if first is None and scores_match and shadow_match else 'mismatch'
        row={'case':case['name'],'policy':policy,'gpu_blocks':blocks,'status':status,
             'native_evictions':len(nd),'v2_evictions':len(cd),'first_difference':first,
             'native_hit_tokens':native['hit_tokens'],'v2_hit_tokens':candidate['hit_tokens'],
             'prompt_tokens':native['prompt_tokens'],'source_sha256':digest(source),
             'scores_match':scores_match,'shadow_checked':shadow is not None,'shadow_match':shadow_match}
    except Exception as error:
        row={'case':case['name'],'policy':policy,'gpu_blocks':blocks,'status':'error','error':str(error)}
    print(f"{row['case']}/{policy}/{blocks}: {row['status']}" + (f" {row.get('first_difference')}" if row['status']=='mismatch' else ''),flush=True)
    (out/case['name']/policy/'comparison.json').write_text(json.dumps(row,indent=2))
    return row


def mixed_case(out):
    # Handwritten, deterministic prefixes exercise task, ghost and reuse branches.
    # These are validation fixtures, not paper traces or held-out workload scores.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/Llama-3.2-1B-Instruct'), local_files_only=True)
    records=[]
    prefixes=['ShareGPT-multi-turn_test','alfred_test','mashqa_test','apps_test',
              'image_ShareGPT-multi-turn_test','file_ShareGPT-multi-turn_test',
              'search_ShareGPT-multi-turn_test']
    for i in range(100):
        sid=f'{prefixes[i%len(prefixes)]}_{i}'
        text1=(f'shared prefix group {i%11} ' * 32) + (f'unique request {i} ' * 16)
        text2=text1 + ' assistant response ' * 8 + (f'followup {i%13} ' * 12)
        response=[9000+(i%7)]*16
        turn1={'input_ids':tokenizer.encode(text1,add_special_tokens=True),
               'response_ids':response,'input_text':text1,'response_text':'synthetic validation response'}
        turn2={'input_ids':tokenizer.encode(text2,add_special_tokens=True),
               'response_ids':response,'input_text':text2,'response_text':'synthetic validation response'}
        records.append({sid:{'id':sid,'num_turns':2,'turn_1':turn1,'turn_2':turn2}})
    data=out/'mixed.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    cfg=out/'mixed.yaml'
    cfg.write_text(yaml.safe_dump({'tasks':[{'name':'mixed','task_id':0,'data_path':str(data),
                    'lambda_rate':1,'mu':0,'sigma':0,'seed':42,'max_record':len(records),'max_trace_duration':-1}]}))
    expected={f'id_{sid}_turn_{t}':(len(d[f'turn_{t}']['input_ids']),len(d[f'turn_{t}']['response_ids']))
              for row in records for sid,d in row.items() for t in (1,2)}
    return {'name':'mixed','config':cfg,'expected':expected}


def snapshot_cases(snapshot, out, names):
    """Relocate frozen records without downloading, reshuffling or retokenizing."""
    cases=[]
    for name in names:
        folder=snapshot/name
        config=yaml.safe_load((folder/'replay.yaml').read_text())
        destination=out/'inputs'/name
        destination.mkdir(parents=True)
        expected={}
        for i,task in enumerate(config['tasks']):
            source=folder/f'data-{i}.jsonl'
            target=destination/f'data-{i}.jsonl'
            shutil.copyfile(source,target)
            task['data_path']=str(target.resolve())
            for line in target.read_text().splitlines():
                for sid,record in json.loads(line).items():
                    for turn in range(1,record['num_turns']+1):
                        data=record[f'turn_{turn}']
                        expected[f'id_{sid}_turn_{turn}']=(len(data['input_ids']),len(data['response_ids']))
        cfg=destination/'replay.yaml'
        cfg.write_text(yaml.safe_dump(config))
        cases.append({'name':name,'config':cfg,'expected':expected})
    return cases


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policies',nargs='+',default=list(POLICIES),choices=POLICIES)
    ap.add_argument('--traces',nargs='*',default=['sharegpt','mashqa','agentbank-apps','alfred','codeparrot'])
    ap.add_argument('--sessions',type=int,default=64)
    ap.add_argument('--gpu-blocks',type=int,default=4096)
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--max-num-seqs',type=int,default=1)
    ap.add_argument('--v2-host-policy',default='LRU',choices=['LRU'])
    ap.add_argument('--output',type=Path)
    ap.add_argument('--no-mixed',action='store_true')
    ap.add_argument('--shadow',action='store_true',help='Also validate private candidate against native events and decisions')
    ap.add_argument('--input-snapshot',type=Path,help='Existing run or its inputs/ directory; reuse exact records, sources and settings')
    args=ap.parse_args()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    snapshot=None
    archived=None
    if args.input_snapshot:
        snapshot=args.input_snapshot.resolve()
        if (snapshot/'inputs').is_dir():snapshot=snapshot/'inputs'
        archived=json.loads((snapshot.parent/'manifest.json').read_text())
        for flag,field in [('--gpu-blocks','gpu_blocks'),('--max-num-seqs','max_num_seqs'),('--policies','policies')]:
            if flag not in sys.argv and field in archived:setattr(args,field,archived[field])
        if '--shadow' not in sys.argv:args.shadow='--shadow' in archived['command']
    out=(args.output or ROOT/'result/policy-interface-v2'/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d/%H%M%S')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'sources').mkdir()
    for policy in args.policies:
        if snapshot:
            original=snapshot.parent/'sources'/f'{policy}.py'
            (out/'sources'/f'{policy}.py').write_bytes(original.read_bytes())
        else:
            (out/'sources'/f'{policy}.py').write_text(source_for_policy(policy))
    cases=[]
    if snapshot:
        names=[c['name'] for c in archived['cases']]
        if '--traces' in sys.argv:
            names=[n for n in names if n in args.traces or (n=='mixed' and not args.no_mixed)]
        if args.no_mixed:names=[n for n in names if n!='mixed']
        cases=snapshot_cases(snapshot,out,names)
    else:
        for trace in args.traces:
            cfgs,expected,manifest=prepare(argparse.Namespace(trace=trace,sessions=args.sessions))
            cases.append({'name':trace,'config':cfgs['tune'],'expected':expected['tune']})
            (out/f'{trace}-inputs.json').write_text(json.dumps(manifest,indent=2))
        if not args.no_mixed:
            cases.append(mixed_case(out))
    for case in cases:
        case['max_num_seqs']=args.max_num_seqs
        case['v2_host_policy']=args.v2_host_policy
        case['shadow']=args.shadow
    codepaths=[*ROOT.glob('policy_agent/*.py'),ROOT/'scripts/policy_interface_v2_parity.py',
               ROOT/'vllm/core/evictor.py',ROOT/'vllm/core/block/prefix_caching_block.py',
               ROOT/'vllm/core/block_manager.py',ROOT/'vllm/core/scheduler.py',
               ROOT/'vllm/engine/llm_engine.py',ROOT/'vllm/simulator/output_simulator.py']
    modelpaths=[p for p in (ROOT/'models/Llama-3.2-1B-Instruct').iterdir() if p.is_file()]
    manifest={'input_snapshot':str(snapshot) if snapshot else None,
              'reference_runtime_revision':archived.get('git_head') if archived else None,
              'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in codepaths},
              'model_sha256':{str(p.relative_to(ROOT)):digest(p) for p in modelpaths},
              'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','transformers')},
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'command':sys.argv,'python':sys.version,'policies':args.policies,
              'cases':[{'name':c['name'],'config':str(c['config']),'sha256':digest(c['config'])} for c in cases],
              'gpu_blocks':args.gpu_blocks,'mixed_gpu_blocks':128,'max_num_seqs':args.max_num_seqs,'v2_host_policy':args.v2_host_policy}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(policy,case,out,128 if case['name']=='mixed' else args.gpu_blocks) for case in cases for policy in args.policies]
    rows=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(lambda x:compare(*x),tasks):
            rows.append(row)
            (out/'results.json').write_text(json.dumps(rows,indent=2))
    print('Results:',out,flush=True)
    lines=['# Native / standalone v2 parity', '', '| Case | Policy | Blocks | Native evictions | V2 evictions | Result |', '|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['policy']} | {r['gpu_blocks']} | {r.get('native_evictions','')} | {r.get('v2_evictions','')} | {r['status']} |")
    lines += ['', 'Every v2 replay uses the fixed LRU host configuration. Only policy.py changes.',
              'Matching includes the entire eviction ledger, trusted hit-token counts, completed request IDs and token lengths.',
              'Synthetic mixed fixtures are validation cases, not paper workload results.',
              '', 'Reproduce using command arguments in manifest.json. Sources and hashes are archived alongside results.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if any(row['status']!='ok' for row in rows):
        raise SystemExit(1)

if __name__=='__main__':
    main()
