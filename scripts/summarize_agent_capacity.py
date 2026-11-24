"""Average saved eval results across workloads at each KV capacity (stdlib only)."""
import argparse
import csv
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    with (args.study / 'agent-vs-opt-eval.csv').open(newline='') as f:
        source = list(csv.DictReader(f))
    results = []
    for capacity in sorted({float(r['modeled_kv_gib']) for r in source}):
        # Matched workload set for all metrics: exclude missing agent results.
        rows = [r for r in source if float(r['modeled_kv_gib']) == capacity
                and r['agent_hit_ratio'] != '']
        if not rows:
            continue
        average = lambda key: mean(float(r[key]) for r in rows)
        gain = average('Agent_minus_LRU_pp')
        gap = average('OPT_minus_LRU_pp')
        eligible = [float(r['headroom_attained_percent']) for r in rows
                    if r['headroom_attained_percent'] != '']
        results.append({
            'modeled_kv_gib': capacity,
            'completed_workloads': len(rows),
            'mean_lru_hit_ratio': average('lru_hit_ratio'),
            'mean_agent_hit_ratio': average('agent_hit_ratio'),
            'mean_change_pp': gain,
            'mean_OPT_minus_LRU_pp': gap,
            'mean_OPT_minus_Agent_pp': average('OPT_minus_Agent_pp_remaining'),
            'mean_headroom_attainment_percent': mean(eligible) if eligible else None,
            'headroom_attainment_workloads': len(eligible),
            'headroom_weighted_attainment_percent': 100 * gain / gap if gap > 0 else None,
        })
    if not results:
        raise ValueError('No completed agent results')
    csv_path = args.study / 'agent-capacity-averages.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ['# Agent effectiveness by modeled KV capacity', '',
             'Eval results. Columns use unweighted workload means over completed agent cases at each capacity. '
             'The same workloads are used for LRU, Agent, change and headroom. '
             '4 GiB excludes the timed-out ALFRED search; other capacities include all five workloads.', '',
             'Mean attainment averages each case\'s (Agent − LRU)/(OPT − LRU), excluding zero-headroom cases. '
             'Headroom-weighted attainment equals mean change / mean headroom; zero-headroom cases have zero weight. '
             'Negative gains are retained. These percentages are relative to the measured offline OPT reference.', '',
             '| KV capacity (GiB) | Workloads | LRU | Agent | Change (pp) | OPT − LRU (pp) | OPT − Agent (pp) | Mean attainment | Headroom-weighted attainment |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        pct = lambda key: f'{r[key]:.2f}%' if r[key] is not None else 'n/a'
        lines.append(f'| {r["modeled_kv_gib"]:g} | {r["completed_workloads"]} '
                     f'| {100*r["mean_lru_hit_ratio"]:.2f}% | {100*r["mean_agent_hit_ratio"]:.2f}% '
                     f'| {r["mean_change_pp"]:+.3f} | {r["mean_OPT_minus_LRU_pp"]:.3f} '
                     f'| {r["mean_OPT_minus_Agent_pp"]:.3f} '
                     f'| {pct("mean_headroom_attainment_percent")} '
                     f'| {pct("headroom_weighted_attainment_percent")} |')
    lines += ['', 'At 24 GiB, mean attainment covers three workloads because ShareGPT and ALFRED already match OPT.', '',
              'Reproduce from the experiment worktree:', '',
              '```bash', f'python3 scripts/summarize_agent_capacity.py --study {args.study}', '```', '']
    (args.study / 'agent-capacity-averages.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
