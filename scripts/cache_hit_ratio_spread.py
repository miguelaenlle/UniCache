"""Print/export the spread between simple policies at each cache capacity.

Uses only Python's standard library; no simulator or API calls.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--split', choices=['tune', 'eval'], default='eval')
    args = parser.parse_args()
    policies = ['LRU', 'LFU', 'FIFO']
    traces = ['sharegpt', 'mashqa', 'agentbank-apps', 'alfred', 'codeparrot']
    labels = ['ShareGPT', 'MASH-QA', 'AgentBank APPS', 'ALFRED', 'CodeParrot']
    data = {}
    with (args.study / 'scores.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] != args.split or r['policy'] not in policies:
                continue
            key = (r['policy'], r['trace'], int(r['gpu_blocks']))
            if key in data:
                raise ValueError(f'Duplicate score: {key}')
            value = float(r['hit_ratio'])
            if not 0 <= value <= 1:
                raise ValueError(f'Invalid score: {key}')
            data[key] = value
    capacity = json.loads((args.study / 'capacity.json').read_text())
    budgets = sorted(int(b) for b in capacity['budgets'])
    rows = []
    for budget in budgets:
        cells = []
        spreads = []
        for trace in traces:
            # Missing cells fail explicitly; compare all three policies at one capacity.
            values = [data[policy, trace, budget] for policy in policies]
            lo, hi = min(values), max(values)
            spreads.append(100 * (hi - lo))
            cells.append(f'{100 * (hi - lo):.2f} pp ({100 * lo:.2f}%–{100 * hi:.2f}%)')
        gib = capacity['budgets'][str(budget)]['modeled_kv_gib']
        rows.append([f'{gib:.3f}'.rstrip('0').rstrip('.') + ' GiB', *cells,
                     f'{sum(spreads) / len(spreads):.2f} pp'])
    capacities = [f'{capacity["budgets"][str(b)]["modeled_kv_gib"]:.3f}'.rstrip('0').rstrip('.')
                  for b in budgets]
    lines = ['# Spread between simple baselines at each KV capacity', '',
             f'{args.split.capitalize()} token-weighted cache hit ratio. Capacities: '
             + ', '.join(capacities) + ' GiB.', '',
             'Cell = highest − lowest hit ratio among LRU, LFU and FIFO at the same capacity, '
             'in percentage points (minimum%–maximum%). '
             'Calculated from unrounded scores; this is a policy spread, not an uncertainty interval.', '',
             'Average is the unweighted mean of the five workload spreads, calculated before rounding.', '',
             '| Modeled KV capacity | ' + ' | '.join(labels) + ' | Average |',
             '|---|' + '---:|' * (len(traces) + 1),
             *['| ' + ' | '.join(row) + ' |' for row in rows], '']
    output = args.study / 'figures'
    output.mkdir(exist_ok=True)
    stem = f'simple-baseline-capacity-spread-{args.split}'
    (output / (stem + '.md')).write_text('\n'.join(lines))
    with (output / (stem + '.csv')).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Modeled KV capacity', *labels, 'Average'])
        writer.writerows(rows)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
