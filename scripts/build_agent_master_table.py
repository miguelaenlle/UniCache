"""Merge saved agent comparisons and summarize a tune-headroom subset (stdlib only)."""
import argparse
import csv
from pathlib import Path
from statistics import mean

LABELS = {'sharegpt': 'ShareGPT', 'mashqa': 'MASH-QA', 'agentbank-apps': 'AgentBank APPS',
          'alfred': 'ALFRED', 'codeparrot': 'CodeParrot'}
COLUMNS = [
    ('modeled_kv_gib', 'KV capacity (GiB)', 'capacity'), ('trace', 'Trace', 'trace'),
    ('best_heuristic_selected_on_tune', 'Best heuristic (tune-selected)', 'text'),
    ('lru_tune_hit_ratio', 'LRU tune (%)', 'ratio'),
    ('agent_tune_hit_ratio', 'Agent tune (%)', 'ratio'),
    ('Agent_minus_LRU_tune_pp', 'Change tune (pp)', 'signed'),
    ('OPT_minus_LRU_tune_pp', 'OPT − LRU tune (pp)', 'number'),
    ('OPT_minus_Agent_tune_pp', 'OPT − Agent tune (pp)', 'number'),
    ('Agent_minus_Best_tune_pp', 'Agent − Best tune (pp)', 'signed'),
    ('lru_hit_ratio', 'LRU eval (%)', 'ratio'),
    ('agent_hit_ratio', 'Agent eval (%)', 'ratio'),
    ('Agent_minus_LRU_pp', 'Change eval (pp)', 'signed'),
    ('OPT_minus_LRU_pp', 'OPT − LRU eval (pp)', 'number'),
    ('OPT_minus_Agent_pp_remaining', 'OPT − Agent eval (pp)', 'number'),
    ('Agent_minus_Best_eval_pp', 'Agent − Best eval (pp)', 'signed'),
    ('headroom_attained_percent', 'Eval headroom attained (%)', 'percent'),
]


def read_csv(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    keys = [(float(r['modeled_kv_gib']), r['trace']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate case: {path}')
    return rows


def formatted(row, field, kind):
    raw = row[field]
    if raw == '':
        return 'n/a' if field == 'headroom_attained_percent' and row['agent_hit_ratio'] else '—'
    if kind == 'text': return raw
    if kind == 'trace': return LABELS[raw]
    v = float(raw)
    if kind == 'capacity': return f'{v:.3f}'.rstrip('0').rstrip('.')
    if kind == 'ratio': return f'{100*v:.2f}'
    if kind == 'percent': return f'{v:.1f}'
    if kind == 'signed': return f'{v:+.3f}'
    return f'{v:.3f}'


def table(rows):
    return '\n'.join([
        '| ' + ' | '.join(label for _, label, _ in COLUMNS) + ' |',
        '|' + '|'.join('---' if kind in ['trace', 'text'] else '---:' for _, _, kind in COLUMNS) + '|',
        *['| ' + ' | '.join(formatted(row, field, kind) for field, _, kind in COLUMNS) + ' |'
          for row in rows],
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--min-tune-headroom-pp', type=float, default=10)
    args = p.parse_args()
    opt = read_csv(args.study / 'agent-vs-opt-eval.csv')
    heuristics = read_csv(args.study / 'agent-vs-best-heuristic.csv')
    by_case = {(float(r['modeled_kv_gib']), r['trace']): r for r in heuristics}
    if {(float(r['modeled_kv_gib']), r['trace']) for r in opt} != set(by_case):
        raise ValueError('Input tables cover different cases')
    rows = []
    for original in opt:
        h = by_case[float(original['modeled_kv_gib']), original['trace']]
        # Fields shared between exports must agree before joining.
        for key in original.keys() & h.keys():
            if original[key] != h[key]:
                raise ValueError(f'Conflicting value: {original["trace"]}/{key}')
        if original['agent_hit_ratio'] != h['agent_eval_hit_ratio'] or original['lru_hit_ratio'] != h['lru_eval_hit_ratio']:
            raise ValueError('Eval scores disagree between exports')
        row = {**original, **h}
        for split in ['tune', 'eval']:
            agent = h[f'agent_{split}_hit_ratio']
            row[f'Agent_minus_Best_{split}_pp'] = (
                str(100 * (float(agent) - float(h[f'best_heuristic_{split}_hit_ratio'])))
                if agent else '')
        rows.append(row)
    selected = [r for r in rows if float(r['OPT_minus_LRU_tune_pp']) > args.min_tune_headroom_pp]
    summaries = []
    for name, cases in [('All cases', rows), ('Tune headroom > threshold', selected)]:
        completed = [r for r in cases if r['agent_hit_ratio'] != '']
        summaries.append({
            'subset': name, 'eligible_cases': len(cases), 'completed_cases': len(completed),
            'mean_Agent_minus_LRU_eval_pp': mean(float(r['Agent_minus_LRU_pp']) for r in completed) if completed else None,
            'mean_Agent_minus_Best_tune_pp': mean(float(r['Agent_minus_Best_tune_pp']) for r in completed) if completed else None,
            'mean_Agent_minus_Best_eval_pp': mean(float(r['Agent_minus_Best_eval_pp']) for r in completed) if completed else None,
        })
    for name, cases in [('agent-master-table', rows), ('agent-master-table-high-headroom', selected)]:
        with (args.study / (name + '.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[field for field, _, _ in COLUMNS], extrasaction='ignore')
            writer.writeheader(); writer.writerows(cases)
        (args.study / (name + '.md')).write_text(table(cases) + '\n')
    with (args.study / 'agent-master-averages.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    report = [
        '# Agent policy master tables', '',
        'Capacity: modeled GiB. Hit ratios: %. Differences: percentage points (pp). '
        'Change = Agent − LRU. Agent − Best is positive when the agent is ahead and negative when it trails. '
        'Best heuristic is selected on tune from online baselines, excluding offline OPT and NONE, '
        'then held fixed for eval. Eval headroom attained = 100 × (Agent − LRU)/(OPT − LRU).', '',
        '## All cases', '', table(rows), '',
        f'## Subset: OPT − LRU on tune > {args.min_tune_headroom_pp:g} pp', '', table(selected), '',
        'The strict threshold uses unrounded tune scores only. It selects workload/capacity cases, '
        'not entire workloads. The timed-out ALFRED / 4 GiB case qualifies but has no selected agent result.', '',
        '## Averages', '',
        '| Subset | Eligible cases | Completed cases | Mean change eval (pp) | Mean Agent − Best tune (pp) | Mean Agent − Best eval (pp) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for s in summaries:
        val = lambda key: f'{s[key]:+.3f}' if s[key] is not None else '—'
        report.append(f'| {s["subset"]} | {s["eligible_cases"]} | {s["completed_cases"]} '
                      f'| {val("mean_Agent_minus_LRU_eval_pp")} | {val("mean_Agent_minus_Best_tune_pp")} '
                      f'| {val("mean_Agent_minus_Best_eval_pp")} |')
    report += ['', 'Means are unweighted across completed workload/capacity cases, calculated before rounding. '
               'Missing agent results are excluded; fallback LRU policies and negative changes remain included. '
               'This is a descriptive subset of the existing run; the filter does not rerun or reselect policies.', '',
               'Reproduce with Python standard library only:', '', '```bash',
               f'python3 scripts/build_agent_master_table.py --study {args.study} --min-tune-headroom-pp {args.min_tune_headroom_pp:g}', '```', '']
    path = args.study / 'agent-master-report.md'
    path.write_text('\n'.join(report))
    completed = [r for r in selected if r['agent_hit_ratio'] != '']
    presentation = [f'# Cases with OPT − LRU on tune > {args.min_tune_headroom_pp:g} pp', '',
                    '| Workload | KV capacity (GiB) | Best heuristic (selected on tune) | Eval headroom: OPT − LRU (pp) | Agent − LRU, eval (pp) | Agent − best heuristic, eval (pp) |',
                    '|---|---:|---|---:|---:|---:|']
    for r in completed:
        presentation.append(f'| {LABELS[r["trace"]]} | {formatted(r,"modeled_kv_gib","capacity")} '
                            f'| {r["best_heuristic_selected_on_tune"]} | {float(r["OPT_minus_LRU_pp"]):.2f} '
                            f'| {float(r["Agent_minus_LRU_pp"]):+.2f} | {float(r["Agent_minus_Best_eval_pp"]):+.2f} |')
    if completed:
        presentation.append(f'| **Average** | — | — | **{mean(float(r["OPT_minus_LRU_pp"]) for r in completed):.2f}** '
                            f'| **{mean(float(r["Agent_minus_LRU_pp"]) for r in completed):+.2f}** '
                            f'| **{mean(float(r["Agent_minus_Best_eval_pp"]) for r in completed):+.2f}** |')
    presentation += ['', 'All differences are percentage points (pp). Positive values favor the agent in both comparison columns. '
                     'Baseline identity is selected on tune and held fixed on eval. Means are unweighted over completed cases; '
                     'the timed-out ALFRED / 4 GiB case is excluded.', '']
    (args.study / 'agent-presentation-table.md').write_text('\n'.join(presentation))
    print(f'Master tables and averages: {path}')
    for summary in summaries: print(summary)


if __name__ == '__main__':
    main()
