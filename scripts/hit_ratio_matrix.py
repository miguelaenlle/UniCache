"""Run upstream policies on the five available workloads and pivot hit ratios."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {
    'ShareGPT': 'TaskData/SharedGPT-multi-turn/tasks_config.yaml',
    'MASH-QA': 'TaskData/mashqa_data/tasks_config.yaml',
    'AgentBank-APPS': 'TaskData/AgentBankData/tasks_config.yaml',
    'ALFRED': 'TaskData/AgentBankData/tasks_config_alfred.yaml',
    'CodeParrot': 'TaskData/Codeparrot/tasks_config.yaml',
}
POLICIES = ['LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA', 'OPT', 'SimpleHYBRID']


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_tables(out, workloads, policies, results):
    # Empty means not completed; failed runs are ERR, never zero hit ratio.
    with (out / 'hit_ratio_matrix.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['policy', *workloads])
        for policy in policies:
            row = ['UniCache' if policy == 'SimpleHYBRID' else policy]
            for workload in workloads:
                record = results.get(f'{workload}/{policy}', {})
                row.append(record.get('hit_ratio', 'ERR' if record.get('error') else ''))
            writer.writerow(row)
    lines = ['| Policy | ' + ' | '.join(workloads) + ' |',
             '|---|' + '---:|' * len(workloads)]
    for policy in policies:
        row = ['UniCache' if policy == 'SimpleHYBRID' else policy]
        for workload in workloads:
            record = results.get(f'{workload}/{policy}', {})
            row.append(f"{record['hit_ratio']:.4%}" if 'hit_ratio' in record
                       else ('ERR' if record.get('error') else '—'))
        lines.append('| ' + ' | '.join(row) + ' |')
    (out / 'hit_ratio_matrix.md').write_text('\n'.join(lines) + '\n')
    (out / 'results.json').write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', nargs='+', choices=CATALOG, default=list(CATALOG))
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    parser.add_argument('--gpu-blocks', type=int, default=50000)
    parser.add_argument('--max-records', type=int, help='Cap each upstream session limit')
    parser.add_argument('--ttft', default='3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)')
    parser.add_argument('--tpot', type=float, default=0.01)
    parser.add_argument('--dry-run', action='store_true', help='Inspect inputs; do not run simulations')
    parser.add_argument('--output', help='Fresh output directory')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.gpu_blocks <= 0 or args.max_records is not None and args.max_records <= 0:
        parser.error('Block and session counts must be positive')
    configs = {}
    for name in args.workloads:
        config = yaml.safe_load(Path(CATALOG[name]).read_text())
        for task in config['tasks']:
            if not Path(task['data_path']).is_file():
                parser.error(f"Missing {name} data: {task['data_path']}")
            if args.max_records is not None:
                limit = task.get('max_record', -1)
                task['max_record'] = min(limit, args.max_records) if limit > 0 else args.max_records
        configs[name] = config
        print(f"{name}: {[(t['max_record'], t['data_path']) for t in config['tasks']]}", flush=True)
    print(f'Policies: {args.policies}; blocks: {args.gpu_blocks}; block size: 8', flush=True)
    if args.dry_run:
        return
    out = Path(args.output or f"result/hit-ratio-matrix/{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'arguments': vars(args), 'configs': configs,
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dataset_sha256': {t['data_path']: checksum(t['data_path'])
                           for c in configs.values() for t in c['tasks']},
        'model_metadata_sha256': {str(p): checksum(p)
                                  for p in Path('models/Llama-3.2-1B-Instruct').glob('*.json')},
        'notes': ['ToolBench omitted: authors source requires inaccessible UCSD authentication.',
                  'Common cache capacity across columns; standalone upstream task/session settings retained.',
                  'OPT uses upstream NONE trace collection and prefill-order future-use mapping.',
                  'Exact paper samples/settings unverified; no QTTFT measurement.'],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = {}
    save_tables(out, args.workloads, args.policies, results)
    print(f'Results directory: {out}', flush=True)

    def run_one(workload, policy, config_path, use_map=None):
        directory = out / workload / policy
        directory.mkdir(parents=True)
        cmd = [sys.executable, '-u', 'run_simulation.py',
               '--model', str(ROOT / 'models/Llama-3.2-1B-Instruct'), '--device', 'cpu',
               '--max-num-seqs', '32', '--max-model-len', '131072', '--block-size', '8',
               '--simulator-enable-virtual-time', '--simulator-num-gpu-blocks', str(args.gpu_blocks),
               '--simulator-num-cpu-blocks', '50000', '--eviction-policy', policy,
               '--simulator-ttft', args.ttft, '--simulator-tpot', str(args.tpot),
               '--simulator-output-dir', str(directory), '--simulator-task-config', str(config_path),
               '--log-level', 'WARNING']
        if policy != 'NONE':
            cmd.append('--enable-prefix-caching')
        if use_map is not None:
            cmd += ['--opt-block-use-map-path', str(use_map)]
        env = dict(os.environ, COLLECT_TRACE='1' if policy == 'NONE' else '0')
        (directory / 'command.json').write_text(json.dumps(cmd, indent=2))
        print(f'Running {workload}/{policy}; log: {directory / "run.log"}', flush=True)
        with (directory / 'run.log').open('w') as log:
            status = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError(f'Exit {status.returncode}; see {directory / "run.log"}')
        summaries = list(directory.glob('*/summary.json'))
        if len(summaries) != 1:
            raise RuntimeError(f'Expected one summary under {directory}')
        return summaries[0]

    for workload, config in configs.items():
        folder = out / workload
        folder.mkdir()
        config_path = folder / 'tasks_config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        use_map = None
        profile_error = None
        if 'OPT' in args.policies:
            try:
                profile_summary = run_one(workload, 'NONE', config_path)
                use_map = profile_summary.parent / 'use_map.json'
                if not use_map.is_file():
                    raise RuntimeError('Offline OPT use_map.json missing')
            except RuntimeError as error:
                profile_error = str(error)
        for policy in args.policies:
            key = f'{workload}/{policy}'
            try:
                if policy == 'OPT' and profile_error:
                    raise RuntimeError(f'OPT profile failed: {profile_error}')
                path = run_one(workload, policy, config_path, use_map if policy == 'OPT' else None)
                summary = json.loads(path.read_text())
                hits = summary['hit_rate_summary']
                results[key] = {'hit_ratio': hits['global_hit_rate'], 'requests': hits['total_requests'],
                                'hit_tokens': hits['global_hit_tokens_num'], 'prompt_tokens': hits['global_tokens'],
                                'runtime_seconds': summary['simulation_time'], 'summary': str(path)}
                print(f"  {key}: {hits['global_hit_rate']:.4%}", flush=True)
            except (RuntimeError, KeyError) as error:
                results[key] = {'error': str(error)}
                print(f'  FAILED {key}: {error}', flush=True)
            save_tables(out, args.workloads, args.policies, results)
    print((out / 'hit_ratio_matrix.md').read_text(), flush=True)
    print(f"Matrix: {out / 'hit_ratio_matrix.csv'}", flush=True)
    return int(any('error' in record for record in results.values()))


if __name__ == '__main__':
    sys.exit(main())
