"""Export editable baseline source or replay an edited policy without a coding agent."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_agent.runner import Evaluator, TRACE_CONFIGS, digest, prepare
from policy_agent.v2_policies import POLICIES, source_for_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    export = commands.add_parser('export')
    export.add_argument('--policy', required=True, choices=POLICIES)
    export.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('evaluate')
    run.add_argument('--policy-file', type=Path, required=True)
    run.add_argument('--trace', choices=TRACE_CONFIGS, required=True)
    run.add_argument('--gpu-blocks', type=int, default=32768)
    run.add_argument('--sessions', type=int)
    run.add_argument('--splits', nargs='+', choices=['tune','eval'], default=['tune','eval'])
    run.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.action == 'export':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(source_for_policy(args.policy))
        print(args.output.resolve())
        return
    if args.gpu_blocks < 1:
        p.error('Cache budget must be positive')
    args.policy_interface = 'v2'
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d/%H%M%S-%f')
    out = (args.output or ROOT/'result/manual-policy-v2'/stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = out/'policy.py'
    candidate.write_text(args.policy_file.read_text())
    configs, expected, manifest = prepare(args)
    (out/'manifest.json').write_text(json.dumps({
        'command':sys.argv,'policy_sha256':digest(candidate),'trace':manifest,
        'interface':'v2','gpu_blocks':args.gpu_blocks,'python':sys.version,
        'host_policy':'LRU (physical scaffold only; no native eviction decisions)'}, indent=2))
    evaluator = Evaluator(args,out,configs,expected)
    rows = []
    for split in dict.fromkeys(args.splits):
        score, _ = evaluator.replay('LRU',split,out/split,candidate)
        rows.append({'split':split, **score})
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print(f"{split}: hit ratio {score['token_weighted_hit_ratio']:.4%}", flush=True)
    with (out/'results.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Results:',out)


if __name__ == '__main__':
    main()
