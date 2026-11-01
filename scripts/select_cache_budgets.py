"""Select cache budgets from a completed tune-only pilot using a fixed rule."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def select(pilot, output):
    pilot=Path(pilot).resolve();output=Path(output).resolve()
    done=json.loads((pilot/'completion.json').read_text())
    if not done['passed']:raise ValueError('A complete successful pilot is required')
    manifest=json.loads((pilot/'manifest.json').read_text())
    if manifest['args']['splits']!=['tune']:raise ValueError('Selection requires a tune-only pilot')
    raw=(pilot/'results.json').read_bytes();results=json.loads(raw)
    selections={};commands=[]
    for trace in manifest['traces']:
        candidates=[]
        for blocks in manifest['budgets'].values():
            base=f'{blocks}blocks/{trace}/tune/'
            lru,online,opt=[results[base+p]['hit_ratio'] for p in ['LRU','SimpleHYBRID','OPT']]
            candidates.append({'blocks':blocks,'LRU':lru,'UniCache':online,'OPT':opt,'online_gain_pp':100*(online-lru),'oracle_gap_pp':100*(opt-lru)})
        if max(c['online_gain_pp'] for c in candidates)>0:
            best=max(candidates,key=lambda c:(c['online_gain_pp'],c['oracle_gap_pp'],c['blocks']))
        else:
            best=max(candidates,key=lambda c:(c['oracle_gap_pp'],c['blocks']))
        selections[trace]={'selected':best,'candidates':candidates,'evidence':'demonstrated online gain' if best['online_gain_pp']>0 else 'oracle-only headroom; no demonstrated UniCache gain'}
        commands.append(['bash','scripts/run-cache-budget-sweep.sh','--jobs','4','--blocks',str(best['blocks']),'--traces',trace,'--output',str(output.parent/f'confirm-{trace}')])
    data={'rule':'Maximize UniCache-minus-LRU tune gain, then OPT-minus-LRU tune gap, then larger budget. No eval data used.',
          'pilot':str(pilot),'pilot_results_sha256':hashlib.sha256(raw).hexdigest(),'selections':selections,'confirmation_commands':commands}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Selection output already exists')
    output.write_text(json.dumps(data,indent=2)+'\n')
    script=output.with_suffix('.sh')
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n# Run from the UniCache repository root.\n'+'\n'.join(shlex.join(c) for c in commands)+'\n')
    print(output);print(script)
    for trace,record in selections.items():print(trace,record['selected'],record['evidence'])
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();select(a.pilot,a.output)
