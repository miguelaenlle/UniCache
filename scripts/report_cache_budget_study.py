"""Validate and report the completed headroom pilot and held-out confirmations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def report(study):
    study=Path(study).resolve();selection=json.loads((study/'selection.json').read_text())
    pilot=study/'pilot';raw=(pilot/'results.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=selection['pilot_results_sha256']:raise ValueError('Pilot results changed since budget selection')
    pilot_results=json.loads(raw)
    rows=[];provenance={};checks={};all_scores={}
    for trace,item in selection['selections'].items():
        folder=study/f'confirm-{trace}';done=json.loads((folder/'completion.json').read_text())
        if not done['passed'] or done['completed']!=16:raise ValueError('Incomplete confirmation: '+trace)
        m=json.loads((folder/'manifest.json').read_text())
        blocks=item['selected']['blocks']
        if list(m['budgets'].values())!=[blocks]:raise ValueError('Confirmation budget mismatch')
        results=json.loads((folder/'results.json').read_text());all_scores[trace]=results
        for policy in ['LRU','SimpleHYBRID','OPT']:
            if results[f'{blocks}blocks/{trace}/tune/{policy}']['hit_ratio'] != pilot_results[f'{blocks}blocks/{trace}/tune/{policy}']['hit_ratio']:
                raise ValueError('Tune replication mismatch: '+trace+'/'+policy)
        with (folder/'headroom.csv').open() as f: records=list(csv.DictReader(f))
        for r in records:r['gpu_blocks']=int(r['gpu_blocks'])
        rows.extend(records)
        provenance[trace]={'budget_blocks':blocks,'run':str(folder),'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),'selection_evidence':item['evidence']}
        checks[trace]={'all_16_comparisons_passed':True,'LRU_UniCache_OPT_tune_replication_exact':True}
    columns=list(rows[0])
    for split in ['tune','eval']:
        with (study/f'{split}_headroom.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(r for r in rows if r['split']==split)
    evaluations=[r for r in rows if r['split']=='eval']
    lines=['# 1B headroom experiment','',
           'Smaller virtual KV caches create substantial headroom on some workloads. All gains below are percentage points, with online policies selected on tune only.', '',
           '| Workload | Blocks | LRU eval | Online policy (tune-selected) | Online eval | OPT eval | Online − LRU (pp) | OPT − LRU (pp) |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in evaluations:
        l,b,o=map(float,[r['LRU'],r['best_online_hit_ratio'],r['OPT']])
        lines.append(f"| {r['trace']} | {r['gpu_blocks']:,} | {l:.3%} | {r['best_online_by_tune']} | {b:.3%} | {o:.3%} | {float(r['online_minus_LRU_pp']):+.3f} | {float(r['OPT_minus_LRU_pp']):+.3f} |")
    lines += ['', '## All baseline eval results', '',
              'Columns use the selected cache budget for each workload; these budgets differ.', '',
              '| Policy | '+' | '.join(selection['selections'])+' |', '|---|'+'---:|'*len(selection['selections'])]
    for split in ['tune','eval']:
        with (study/f'baseline_{split}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['policy',*selection['selections']])
            for policy in ['LRU','LFU','FIFO','LeCaR','ARC','WA','SimpleHYBRID','OPT']:
                values=[]
                for trace,item in selection['selections'].items():
                    blocks=item['selected']['blocks']
                    values.append(all_scores[trace][f'{blocks}blocks/{trace}/{split}/{policy}']['hit_ratio'])
                label='UniCache' if policy=='SimpleHYBRID' else policy
                w.writerow([label,*values])
                if split=='eval':lines.append('| '+label+' | '+' | '.join(f'{v:.3%}' for v in values)+' |')
    lines += ['', '## Graphs', '', '![Confirmed eval headroom](headroom.png)', '', '![Tune-only budget pilot](pilot-curves.png)', '']
    reference=study/'reference'
    if reference.exists():
        snapshot=json.loads((reference/'snapshot.json').read_text())
        for name,expected in snapshot['sha256'].items():
            if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=expected:raise ValueError('Reference snapshot changed: '+name)
        with (reference/'headroom.csv').open() as f: reference_rows=[r for r in csv.DictReader(f) if r['split']=='eval' and float(r['budget'])==.5]
        lines += ['', '## Oversized reference (159,125 blocks; modeled 0.5A)', '',
                  '| Trace | LRU eval | OPT eval | OPT − LRU (pp) |','|---|---:|---:|---:|']
        for r in reference_rows:lines.append(f"| {r['trace']} | {float(r['LRU']):.3%} | {float(r['OPT']):.3%} | {float(r['OPT_minus_LRU_pp']):.4f} |")
        lines += ['', 'The remaining oversized reference grid was stopped intentionally. Completed rows are preserved, but incomplete/missing configurations are not presented as measured results.', '']
    decisions=json.loads((study/'decisions.json').read_text())
    lines+=['','## Decisions and controls','']
    lines += ['- '+d['reason'] for d in decisions['decisions']]
    lines += ['- Budget selection used only the complete tune-only pilot. All confirmation eval scores were observed after budget selection.',
              '- Main policies: LRU, LFU, FIFO, LeCaR, ARC, WA, UniCache and offline OPT. No agents or extra models were run.',
              '- Every selected LRU/UniCache/OPT tune score replicated exactly in the fresh confirmation run. All 80 full-policy confirmations passed request/token accounting checks.',
              '- No logged preemption warnings were observed in the pilot; warnings are rate-limited, not exact preemption counts.',
              '', '## Limits', '',
              '- The selector did not read eval data. Prior 50,000-block eval results were already known, so this is not claimed as a new blind holdout study.',
              '- These are exploratory, simulator-based results on one seeded session partition/arrival process, not hardware latency measurements or statistical confidence intervals.',
              '- OPT uses privileged future-use information and is a measured reference, not a guarantee of achievable online improvement.',
              '- The pilot compared LRU, UniCache and OPT at four budgets; selected budgets are best under the declared rule among those tested, not globally optimal.',
              '- Runtime timing uses the existing 1B approximation unchanged. Source subsets are the authors’ standalone limits, not the paper’s mixed Traces #1–3.',
              '- The original A100-normalized reference was intentionally stopped after mostly saturated results; only completed reference measurements are valid. Smaller explicit budgets are not relabelled as 0.5A/0.7A/0.9A.',
              '', '## Reproduce', '',
              'Use EXPERIMENTS.md for the exact reference, pilot, selection and confirmation commands. Archived prepared source prefixes, split IDs/configs/hashes and model metadata are in inputs/ and inputs.json. Source snapshots and environment details are recorded in reproduce.json in each run.',
              '', 'Final report command:', '', '```bash', f'.venv-sim312/bin/python scripts/report_cache_budget_study.py --study {study.relative_to(Path.cwd())}', '```','']
    (study/'report.md').write_text('\n'.join(lines))
    (study/'verification.json').write_text(json.dumps({'checks':checks,'provenance':provenance},indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in [(axes[0],'OPT_minus_LRU_pp','Gap to offline OPT'),(axes[1],'online_minus_LRU_pp','Online gain (policy selected on tune)')]:
        values=[float(r[metric]) for r in evaluations]
        bars=ax.bar(range(len(rows)//2),values)
        ax.set_xticks(range(len(evaluations)),[r['trace']+'\n'+f"{r['gpu_blocks']:,} blocks" for r in evaluations],rotation=20,ha='right',fontsize=8)
        ax.axhline(0,color='black',linewidth=.7);ax.set(title=title,ylabel='Eval gain over LRU (percentage points)')
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.margins(y=.2)
    fig.suptitle('Llama-3.2-1B: confirmed cache-budget headroom',fontsize=12)
    fig.tight_layout();fig.savefig(study/'headroom.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    labels={'LRU':'LRU','SimpleHYBRID':'UniCache','OPT':'Offline OPT'}
    budgets=sorted(next(iter(selection['selections'].values()))['candidates'],key=lambda c:c['blocks'])
    blocks=[c['blocks'] for c in budgets]
    for ax,(trace,item) in zip(axes.flat,selection['selections'].items()):
        for policy,label in labels.items():
            values=[100*pilot_results[f'{b}blocks/{trace}/tune/{policy}']['hit_ratio'] for b in blocks]
            ax.plot(blocks,values,'o-',label=label)
        ax.axvline(item['selected']['blocks'],linestyle='--',color='gray',alpha=.7,label='Selected budget')
        ax.set(title=trace,xlabel='GPU KV blocks (8 tokens/block)',ylabel='Tune hit ratio (%)',ylim=(0,10 if trace=='agentbank-apps' else 100))
        ax.set_xticks(blocks,[f'{b:,}' for b in blocks],rotation=35,ha='right',fontsize=8)
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='upper left',frameon=False)
    axes.flat[-1].text(0,.45,'Tune-only pilot; full source subsets.\nSelection used no eval scores.\nAPPS axis: 0–10%; others: 0–100%.\nOnly measured points are shown.',transform=axes.flat[-1].transAxes,fontsize=10)
    fig.suptitle('Llama-3.2-1B: cache-budget pilot',fontsize=12);fig.tight_layout()
    fig.savefig(study/'pilot-curves.png',dpi=160);plt.close(fig)
    print(study/'report.md')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    report(p.parse_args().study)
