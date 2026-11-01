"""Snapshot explicit policy-agent runs and reproduce cross-workload hit-ratio tables."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

FILES = ['run.json', 'comparison.csv', 'submissions.jsonl', 'selection.json',
         'baseline-results.json', 'stop.json', 'policy.py']
TRACES = ['sharegpt', 'mashqa', 'agentbank-apps', 'alfred', 'codeparrot']
LABELS = dict(zip(TRACES, ['ShareGPT', 'MASH-QA', 'AgentBank APPS', 'ALFRED', 'CodeParrot']))
ORDER = ['LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA', 'UniCache', 'OPT',
         'OPT-Online', 'SimpleHYBRID-OPT', 'OFFSET-HEAD', 'OFFSET-TAIL', 'NONE', 'Agent']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_run(folder):
    data = {name:(folder/name).read_bytes() for name in FILES}
    return data


def validate(data):
    meta = json.loads(data['run.json'])
    args = meta['args']
    stop = json.loads(data['stop.json'])
    if stop.get('exit_code', 0) != 0 or stop.get('reason') in ('agent_error', 'claude_startup_timeout', 'required_baseline_failed'):
        raise ValueError('Run did not complete successfully')
    rows = [json.loads(line) for line in data['submissions.jsonl'].decode().splitlines()]
    valid = [r for r in rows if r['status'] == 'ok']
    if not valid:
        raise ValueError('No successful submission')
    best = max(valid, key=lambda r:(r['tune']['token_weighted_hit_ratio'], -r['submission']))
    if json.loads(data['selection.json'])['submission'] != best['submission']:
        raise ValueError('Selected policy does not match best tune submission')
    if sha(data['policy.py']) != best['policy_sha256']:
        raise ValueError('Final policy hash does not match selected submission')
    table = {}
    for row in csv.DictReader(io.StringIO(data['comparison.csv'].decode())):
        name = row['policy']
        if name in table:
            raise ValueError('Duplicate policy in comparison.csv: '+name)
        metrics = {}
        for split in ['tune', 'eval']:
            raw = row[split+'_hit_ratio']
            value = float(raw) if raw else None
            if value is not None and not 0 <= value <= 1:
                raise ValueError('Invalid hit ratio')
            metrics[split] = value
        table[name] = metrics
        if name == 'Agent':
            if int(row['submission']) != best['submission']:
                raise ValueError('CSV selected submission mismatch')
            for split in ['tune', 'eval']:
                expected = best.get(split, {}).get('token_weighted_hit_ratio')
                if metrics[split] != expected:
                    raise ValueError('CSV agent score mismatch: '+split)
    if 'Agent' not in table:
        raise ValueError('Missing Agent comparison row')
    baselines = json.loads(data['baseline-results.json'])
    errors = {}
    for name, scores in baselines.items():
        if name not in table:
            raise ValueError('Baseline missing from CSV: '+name)
        for split in ['tune', 'eval']:
            expected = scores.get(split, {}).get('token_weighted_hit_ratio')
            if table[name][split] != expected:
                raise ValueError('CSV baseline score mismatch: '+name+'/'+split)
            if split+'_error' in scores:
                errors[name+'/'+split] = scores[split+'_error']
    return {'trace':args['trace'], 'args':args, 'table':table, 'errors':errors,
            'selected':best['submission'], 'attempts':len([r for r in rows if r['submission']>0]),
            'cap':args['max_submissions'], 'stop_reason':stop['reason']}


def summarize(runs=None, manifest=None, output=None):
    # Validate and read everything before creating output. Explicit selection only;
    # never choose the latest directory implicitly.
    sources = []
    if manifest:
        manifest = Path(manifest).resolve()
        recorded = json.loads(manifest.read_text())
        for item in recorded['runs']:
            data = {}
            for name, info in item['files'].items():
                path = (manifest.parent/info['snapshot']).resolve()
                if not path.is_relative_to(manifest.parent):
                    raise ValueError('Snapshot path escapes summary directory')
                raw = path.read_bytes()
                if sha(raw) != info['sha256']:
                    raise ValueError('Snapshot checksum mismatch: '+str(path))
                data[name] = raw
            sources.append((item['source_run'], data))
    else:
        for folder in runs:
            folder = Path(folder).resolve()
            sources.append((str(folder), read_run(folder)))
    parsed = [(source, data, validate(data)) for source, data in sources]
    traces = [item['trace'] for _, _, item in parsed]
    if len(set(traces)) != len(traces):
        raise ValueError('Duplicate workload: select one run per trace explicitly')
    settings = {(r['args']['serving_model'], r['args']['gpu_blocks'], r['args']['start_policy']) for _, _, r in parsed}
    if len(settings) != 1:
        raise ValueError('Runs mix serving models, cache budgets, or starting policies')
    parsed.sort(key=lambda x:TRACES.index(x[2]['trace']) if x[2]['trace'] in TRACES else len(TRACES))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'version':1, 'summarizer_sha256':sha(Path(__file__).read_bytes()), 'runs':[]}
    for source, data, record in parsed:
        entry = {'trace':record['trace'], 'source_run':source, 'files':{}}
        for name, raw in data.items():
            relative = Path('inputs')/record['trace']/name
            path = output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            entry['files'][name] = {'snapshot':str(relative), 'sha256':sha(raw)}
        provenance['runs'].append(entry)
    (output/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    names = set().union(*(r['table'] for _, _, r in parsed))
    policies = [p for p in ORDER if p in names]+sorted(names-set(ORDER))
    labels = [LABELS.get(r['trace'],r['trace']) for _, _, r in parsed]
    md = ['# Policy-agent summary', '',
          'Token-weighted cache hit ratio. Each Agent cell is a separately tuned policy, selected by tune score only.',
          'CSV ratios are on a 0–1 scale; Markdown values are percentages. Blank means unavailable; ERR means recorded failure.', '']
    for split in ['eval','tune']:
        md += ['## '+split.capitalize(), '', '| Policy | '+' | '.join(labels)+' |', '|---|'+'---:|'*len(labels)]
        with (output/(split+'.csv')).open('w',newline='') as f:
            w = csv.writer(f);w.writerow(['policy', *labels])
            for policy in policies:
                cells=[];display=[]
                for _, _, r in parsed:
                    value=r['table'].get(policy,{}).get(split)
                    failed = policy+'/'+split in r['errors']
                    if policy=='Agent' and value is None:
                        failed = True
                    cells.append(value if value is not None else ('ERR' if failed else ''))
                    display.append(f'{100*value:.4f}%' if value is not None else ('ERR' if failed else '—'))
                w.writerow([policy,*cells]);md.append('| '+policy+' | '+' | '.join(display)+' |')
        md.append('')
    md += ['## Runs', '', '| Workload | Attempts / cap | Selected submission |', '|---|---:|---:|']
    with (output/'runs.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['trace','attempts','cap','selected_submission','source_run','stop_reason'])
        for source, _, r in parsed:
            w.writerow([r['trace'],r['attempts'],r['cap'],r['selected'],source,r['stop_reason']])
            md.append(f"| {LABELS.get(r['trace'],r['trace'])} | {r['attempts']} / {r['cap']} | {r['selected']} |")
    md += ['', 'OPT has privileged future information. These runs use the starting-policy-dependent event interface.',
           'Stop reasons in runs.csv are the agent’s explanations, not independently established optimality claims.', '']
    (output/'summary.md').write_text('\n'.join(md))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--runs',nargs='+',type=Path,help='Explicit completed run directories, one per trace')
    group.add_argument('--manifest',type=Path,help='Rebuild from a prior summary’s verified snapshots')
    p.add_argument('--output',type=Path,default=Path('result/policy-agent-summary')/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d/%H%M%S-%f'))
    args=p.parse_args()
    try:
        output=summarize(args.runs,args.manifest,args.output)
    except (OSError,ValueError,KeyError) as error:
        p.exit(1,str(error)+'\n')
    print('Summary:',output)
    print('Eval:',output/'eval.csv')
    print('Readable tables:',output/'summary.md')

if __name__=='__main__':
    main()
