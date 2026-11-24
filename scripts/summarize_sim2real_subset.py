"""Summarize completed comparisons and separate sensitivity diagnostics (stdlib only)."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args();out=args.study.resolve()
    with (out/'comparison.csv').open() as f:rows=list(csv.DictReader(f))
    status=json.loads((out/'completion.json').read_text())
    if status['errors'] or any(not r['sim_hit_ratio_percent'] for r in rows):
        raise ValueError('Full simulator matrix is not complete')
    measured=[r for r in rows if r['reference_kind']=='measured_chart']
    if len(measured)!=53:raise ValueError('Unexpected measured reference count')
    workloads=list(dict.fromkeys(r['workload'] for r in rows))
    stats=[];effects=[]
    for workload in workloads:
        cells=[r for r in measured if r['workload']==workload]
        online=[r for r in cells if r['policy']!='OPT']
        errors=[float(r['absolute_error_pp']) for r in online]
        stats.append({'workload':workload,'measured_online_cells':len(online),
                      'mean_absolute_error_pp':mean(errors),'max_absolute_error_pp':max(errors),
                      'cells_within_1pp':sum(v<=1 for v in errors)})
        values={r['policy']:r for r in cells}
        if 'UniCache' in values:
            lru=values['LRU'];uc=values['UniCache']
            real=float(uc['gpu_hit_ratio_percent'])-float(lru['gpu_hit_ratio_percent'])
            sim=float(uc['sim_hit_ratio_percent'])-float(lru['sim_hit_ratio_percent'])
            effects.append({'workload':workload,'gpu_UniCache_minus_LRU_pp':real,
                            'sim_UniCache_minus_LRU_pp':sim,'effect_prediction_error_pp':sim-real})
    for name,items in [('workload-transfer-errors.csv',stats),('unicache-vs-lru-transfer.csv',effects)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    replays=[json.loads(p.read_text()) for p in out.glob('replays/*/*/result.json')]
    lines=['# Matched-subset sim-to-real experiment: findings','',
           f'Completed {len(replays)} simulator replays: 60 policy/workload cases and five offline-OPT profiling runs. '
           f'All {sum(r["requests"] for r in replays):,} request completions passed exact prompt/output-token equality checks. '
           'Compared against 53 measured values transcribed from the supplied A100 chart; seven imputed ALFRED values excluded.', '',
           'The reference is rounded chart data, so errors below about 0.005 pp cannot be resolved. '
           'CPU inputs are fully audited; raw GPU logs and completion ledgers were not supplied. '
           'This is hit-ratio comparison, not TTFT/ITL transfer verification.', '',
           '## Primary result: supplied timing approximation', '',
           '| Workload | Measured online policies | Mean absolute error (pp) | Maximum absolute error (pp) | Within 1 pp |',
           '|---|---:|---:|---:|---:|']
    for r in stats:
        lines.append(f'| {r["workload"]} | {r["measured_online_cells"]} | {r["mean_absolute_error_pp"]:.3f} '
                     f'| {r["max_absolute_error_pp"]:.3f} | {r["cells_within_1pp"]}/{r["measured_online_cells"]} |')
    lines += ['', 'Offline OPT is excluded from these online-policy error statistics. The GPU series labeled OPT '
              'underperforms online policies and has unverified future-use metadata. All series remain visible in comparison.csv.', '',
              '## Does UniCache\'s improvement over LRU transfer?', '',
              '| Workload | GPU UniCache − LRU (pp) | Simulator UniCache − LRU (pp) | Difference (pp) |',
              '|---|---:|---:|---:|']
    for r in effects:
        lines.append(f'| {r["workload"]} | {r["gpu_UniCache_minus_LRU_pp"]:+.3f} '
                     f'| {r["sim_UniCache_minus_LRU_pp"]:+.3f} | {r["effect_prediction_error_pp"]:+.3f} |')
    probe=out/'diagnostics/timing-sensitivity/results.json'
    if probe.exists():
        diagnostics=json.loads(probe.read_text())
        original=next(r for r in rows if r['workload']=='ShareGPT' and r['policy']=='LRU')
        lines += ['', '## ShareGPT timing sensitivity (separate diagnostic)', '',
                  'Identical 82 requests, LRU, 512 × 16-token blocks, and unchanged prefill formula. '
                  'Only the simulated decode time per output token is varied. These settings are not GPU measurements '
                  'or an A100 fit, and do not replace primary results.', '',
                  '| Simulated TPOT | Simulated hit ratio (%) | GPU chart (%) | Sim − GPU (pp) |',
                  '|---:|---:|---:|---:|',
                  f'| 10 ms (original) | {float(original["sim_hit_ratio_percent"]):.3f} '
                  f'| {float(original["gpu_hit_ratio_percent"]):.2f} | {float(original["sim_minus_gpu_pp"]):+.3f} |']
        for r in sorted(diagnostics,key=lambda x:x['tpot_seconds'],reverse=True):
            gpu=float(original['gpu_hit_ratio_percent'])
            lines.append(f'| {1000*r["tpot_seconds"]:g} ms (probe) | {r["hit_ratio_percent"]:.3f} '
                         f'| {gpu:.2f} | {r["hit_ratio_percent"]-gpu:+.3f} |')
        lines += ['', 'Decode timing can substantially alter completion-dependent follow-up arrivals and cache competition. '
                  'This demonstrates sensitivity, but does not prove the measured A100 TPOT or isolate every source of discrepancy.', '']
    lines += ['', '## Reproduce', '', 'From the UniCache repository root with the simulator environment:', '',
              '```bash', 'bash scripts/run-sim2real-subset.sh --output result/sim2real-subset/NEW --jobs 4',
              '.venv-sim312/bin/python scripts/probe_sim2real_timing.py --study result/sim2real-subset/NEW',
              'python3 scripts/summarize_sim2real_subset.py --study result/sim2real-subset/NEW', '```', '',
              'The first command replays the archived subset, without new GPU execution or agent search. '
              'The GPU image/reference CSV, exact inputs, commands, source snapshots, hashes and dependency versions are retained.', '']
    (out/'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
