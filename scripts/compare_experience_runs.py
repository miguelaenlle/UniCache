"""Compare policy-agent runs with and without --experience.

Runs are grouped by configuration (trace, serving model, interface, GPU blocks,
sessions, start policy, submission budget). Within a group, runs without
--experience are the control arm and runs with it are the treatment arm, ordered
by start time so the treatment's chain position is visible. Hit ratios are
reported in pp for tune and eval separately, against LRU and against the
best-heuristic selector (best of LRU, LFU, FIFO, LeCaR and UniCache on tune).

    .venv-sim312/bin/python scripts/compare_experience_runs.py --since 2026-11-27
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
HEURISTICS = ['LRU', 'LFU', 'FIFO', 'LeCaR', 'UniCache']
SPLITS = ['tune', 'eval']
FIELDS = ['group', 'arm', 'chain', 'run', 'submissions', 'agent_tune', 'agent_eval',
          'best_heuristic', 'vs_lru_tune_pp', 'vs_lru_eval_pp', 'vs_best_tune_pp',
          'vs_best_eval_pp', 'first_submission_beating_best_tune']


def pp(a, b):
    return None if a is None or b is None else 100 * (a - b)


def read_run(folder):
    """Return one run's summary, or None for prepare-only or unfinished runs."""
    if not (folder/'run.json').is_file() or not (folder/'comparison.csv').is_file():
        return None
    args = json.loads((folder/'run.json').read_text())['args']
    if args.get('prepare_only'):
        return None
    table = {}
    with (folder/'comparison.csv').open() as f:
        for row in csv.DictReader(f):
            table[row['policy']] = {s: float(row[s+'_hit_ratio']) if row[s+'_hit_ratio'] else None
                                    for s in SPLITS}
    if 'Agent' not in table:
        return None
    present = [h for h in HEURISTICS if table.get(h, {}).get('tune') is not None]
    best = max(present, key=lambda h: table[h]['tune']) if present else None
    subs = []
    if (folder/'submissions.jsonl').is_file():
        subs = [json.loads(line) for line in (folder/'submissions.jsonl').read_text().splitlines() if line]
    first = None
    if best:
        first = next((r['submission'] for r in subs if r.get('status') == 'ok'
                      and r['tune']['token_weighted_hit_ratio'] > table[best]['tune']), None)
    agent, lru = table['Agent'], table.get('LRU', {})
    ref = table[best] if best else {}
    group = (f"{args['trace']}/{args['serving_model']}/{args.get('policy_interface', 'v1')}/"
             f"{args['gpu_blocks']}blocks/{args.get('sessions') or 'default'}sessions/"
             f"start={args['start_policy']}/budget={args['max_submissions']}")
    return {'group': group, 'arm': 'experience' if args.get('experience') else 'control',
            'run': folder.relative_to(folder.parents[1]).as_posix(), 'submissions': len(subs),
            'agent_tune': agent['tune'], 'agent_eval': agent['eval'], 'best_heuristic': best,
            **{f'vs_lru_{s}_pp': pp(agent[s], lru.get(s)) for s in SPLITS},
            **{f'vs_best_{s}_pp': pp(agent[s], ref.get(s)) for s in SPLITS},
            'first_submission_beating_best_tune': first}


def collect(root, since=None):
    runs = []
    for folder in sorted(p.parent for p in root.glob('*/*/run.json')):
        if since and folder.parent.name < since:
            continue
        run = read_run(folder)
        if run:
            runs.append(run)
    chains = {}
    for run in runs:  # sorted by start time, so chain numbers follow run order
        key = (run['group'], run['arm'])
        chains[key] = chains.get(key, 0) + 1
        run['chain'] = chains[key]
    return runs


def fmt(value, digits=2):
    if value is None:
        return '-'
    return f'{value:+.{digits}f}' if isinstance(value, float) else str(value)


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    return f'{mean(values):+.2f}' + (f' ± {stdev(values):.2f}' if len(values) > 1 else '') + f' (n={len(values)})'


def report(runs):
    lines = []
    for group in dict.fromkeys(r['group'] for r in runs):
        members = [r for r in runs if r['group'] == group]
        lines += ['', f'## {group}', '',
                  '| arm | # | run | subs | tune vs LRU | eval vs LRU | tune vs best | eval vs best | best heuristic | first sub > best (tune) |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in members:
            lines.append(f"| {r['arm']} | {r['chain']} | {r['run']} | {r['submissions']} | "
                         f"{fmt(r['vs_lru_tune_pp'])} | {fmt(r['vs_lru_eval_pp'])} | "
                         f"{fmt(r['vs_best_tune_pp'])} | {fmt(r['vs_best_eval_pp'])} | "
                         f"{r['best_heuristic'] or '-'} | {fmt(r['first_submission_beating_best_tune'])} |")
        lines += ['', 'Mean pp ± sd by arm:', '']
        for arm in ['control', 'experience']:
            sel = [r for r in members if r['arm'] == arm]
            if sel:
                lines.append(f"* {arm}: " + ', '.join(f"{k.replace('_pp', '').replace('_', ' ')} {stat([r[k] for r in sel])}"
                                                      for k in ['vs_lru_tune_pp', 'vs_lru_eval_pp',
                                                                'vs_best_tune_pp', 'vs_best_eval_pp']))
        chain = [r for r in members if r['arm'] == 'experience']
        if len(chain) > 1:
            lines.append('* experience chain, eval vs best by position: '
                         + ' → '.join(fmt(r['vs_best_eval_pp']) for r in chain))
    return '\n'.join(lines).lstrip('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT/'result/policy-agent')
    p.add_argument('--since', help='Only include run dates on or after YYYY-MM-DD')
    p.add_argument('--csv', type=Path, help='Also write one row per run to this CSV')
    args = p.parse_args()
    runs = collect(args.root, args.since)
    if not runs:
        raise SystemExit(f'No finished policy-agent runs under {args.root}')
    print(report(runs))
    if args.csv:
        with args.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(runs)


if __name__ == '__main__':
    main()
