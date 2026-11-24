"""Replay the exact friend's GPU subsets and compare against a chart transcription.

The GPU reference is rounded chart data, not raw measurement logs. Estimated
ALFRED cells are excluded from all comparison statistics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ['LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA', 'UniCache', 'OPT',
            'OPT-Online', 'SimpleHYBRID-OPT', 'OFFSET-HEAD', 'OFFSET-TAIL']
FOLDERS = {'ShareGPT': 'demo-blocks512', 'ALFRED': 'alfred-2sessions-smallcache-blocks512',
           'CodeParrot': 'codeparrot-100requests-smallcache-blocks512',
           'AgentBank APPS': 'agentbank-apps-selection-pool-blocks512',
           'MASH-QA': 'mashqa-100requests-smallcache-blocks512'}
GPU_CHART = {
    'ShareGPT': [90.57, 90.65, 93.04, 90.52, 90.60, 90.79, 90.47, 90.47, 90.52, 90.47, 93.04, 90.76],
    'CodeParrot': [45.55, 62.55, 43.55, 56.19, 45.24, 45.55, 64.75, 45.55, 45.52, 64.75, 22.76, 63.30],
    'AgentBank APPS': [5.04, 6.09, 5.00, 5.36, 5.24, 5.04, 5.73, 5.04, 5.04, 5.73, 4.75, 5.73],
    'MASH-QA': [27.60, 31.50, 26.01, 32.78, 27.17, 26.43, 31.52, 27.25, 26.40, 31.66, 18.25, 31.77],
    'ALFRED': [97.33] * 12,
}
DEFAULT_IMAGE = Path('/var/folders/6w/gdkc6rf562l823b6xhg5lg_80000gn/T/codex-clipboard-1517bd25-c4e9-4c57-ae40-1b651642037f.png')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def prepare(bundle, out, image):
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        if digest(bundle / name.lstrip('*')) != sha:
            raise ValueError('Bundle checksum mismatch: ' + name)
    manifest = {'hardware_reported_by_user': 'one A100 worker; tensor parallel degree 1',
                'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'python': sys.version, 'bundle': str(bundle),
                'comparison': 'global token-weighted prefix cache hit ratio; simulation minus GPU in pp',
                'gpu_reference': 'User-provided chart, rounded to 0.01 percentage points; no raw GPU logs',
                'reference_rounding_half_width_pp': .005,
                'timing': 'Bundled starter TTFT/TPOT approximation, not newly calibrated to A100',
                'workloads': {}, 'model_metadata_sha256': {}, 'code_sha256': {}}
    manifest['dependency_versions'] = {}
    for package in ['torch','numpy','scipy','transformers','tokenizers','matplotlib','PyYAML']:
        manifest['dependency_versions'][package] = package_metadata.version(package)
    model_meta = json.loads((bundle / 'data/prepared/codeparrot-100requests-smallcache/manifest.json').read_text())
    for name, sha in model_meta['model']['files'].items():
        local = ROOT / 'models/Llama-3.2-1B-Instruct' / name
        if local.exists():
            if digest(local) != sha:
                raise ValueError('Model/tokenizer file differs from GPU manifest: ' + name)
            manifest['model_metadata_sha256'][name] = sha
    manifest['model_metadata_note'] = 'Separate chat_template.jinja not local; tokenizer_config.json matches and includes the chat template.'
    for name in ['run_simulation.py', 'scripts/sim2real_replay.py', 'scripts/sim2real_subset.py',
                 'vllm/core/evictor.py', 'vllm/core/block_manager.py', 'vllm/core/scheduler.py',
                 'vllm/simulator/simulator.py', 'vllm/simulator/output_simulator.py',
                 'vllm/simulator/opt_utils.py', 'vllm/simulator/wa_helper.py']:
        manifest['code_sha256'][name] = digest(ROOT / name)
        snapshot = out / 'code-snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, snapshot)
    if image.exists():
        shutil.copyfile(image, out / 'gpu-reference-chart.png')
        manifest['gpu_reference_image_sha256'] = digest(image)
    with (out / 'gpu-chart-reference.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['workload', 'policy', 'hit_ratio_percent', 'reference_kind'])
        for workload in FOLDERS:
            for policy, value in zip(POLICIES, GPU_CHART[workload]):
                kind = 'estimated_chart' if workload == 'ALFRED' and POLICIES.index(policy) >= 5 else 'measured_chart'
                writer.writerow([workload, policy, value, kind])
    for workload, folder in FOLDERS.items():
        source = bundle / 'data/prepared' / folder
        config = json.loads((source / 'config.json').read_text())
        if (config['gpu_blocks'], config['block_size'], config['max_model_len'], config['max_num_seqs'], config['cpu_blocks'], config['tensor_parallel_size']) != (512, 16, 8192, 32, 0, 1):
            raise ValueError('Unexpected GPU configuration: ' + workload)
        target = out / 'inputs' / folder
        target.mkdir(parents=True)
        shutil.copyfile(source / 'config.json', target / 'original-config.json')
        data_path = target / 'sessions.jsonl'
        source_data = bundle / config['tasks'][0]['data_path']
        shutil.copyfile(source_data, data_path)
        expected = {}
        sessions = 0
        for line in data_path.read_text().splitlines():
            for sid, record in json.loads(line).items():
                if sid != record['id']: raise ValueError('Session ID mismatch')
                sessions += 1
                for turn in range(1, record['num_turns'] + 1):
                    key = f'id_{sid}_turn_{turn}'
                    if key in expected: raise ValueError('Duplicate request')
                    data = record[f'turn_{turn}']
                    expected[key] = {k: data[k] for k in ['input_ids', 'response_ids']}
        if sessions != config['tasks'][0]['max_record']:
            raise ValueError('Session count mismatch')
        write(target / 'expected-tokens.json', expected)
        task = dict(config['tasks'][0], data_path=str(data_path))
        (target / 'replay.yaml').write_text(yaml.safe_dump({'tasks': [task]}, sort_keys=False))
        manifest['workloads'][workload] = {'folder': folder, 'sessions': sessions, 'requests': len(expected),
                                         'prompt_tokens': sum(len(v['input_ids']) for v in expected.values()),
                                         'dataset_sha256': digest(data_path), 'config': config,
                                         'input_files_sha256':{name:digest(target/name) for name in
                                                               ['sessions.jsonl','original-config.json','replay.yaml','expected-tokens.json']}}
    manifest['gpu_reference_csv_sha256'] = digest(out/'gpu-chart-reference.csv')
    write(out / 'manifest.json', manifest)
    (out/'README.md').write_text(
        '# Matched GPU-subset replay\n\n'
        'From the UniCache repository root, use the existing Python 3.12 simulator environment:\n\n'
        '```bash\nbash scripts/run-sim2real-subset.sh --bundle '+str(bundle)+' --output result/sim2real-subset/REPRODUCE-NEW --jobs 4\n```\n\n'
        'This replays all five frozen subsets and 12 chart policies, plus a separate NONE profile for each offline OPT run. '
        'The original JSON configs are retained; only data paths are relocated into YAML for the local loader. '
        'No candidate agent search or new GPU execution is involved.\n\n'
        'Run `bash scripts/run-sim2real-subset.sh --stage compare --output '+str(out)+'` to regenerate tables/graphs without simulations. '
        'The GPU chart transcription is gpu-chart-reference.csv; its estimated ALFRED rows are excluded from errors. '
        'The original image is preserved. Raw GPU logs are unavailable. No latency-transfer claim is made.\n\n'
        'Every completed CPU run includes a command, log, exact token-completion ledger, simulator outputs, and result file. '
        'The manifest records the Git revision, dependency versions, input hashes, and source snapshots. '
        'Do not compare imputed GPU values as measurements. Use a fresh folder for a new experiment.\n')
    return manifest


def replay(out, workload, policy, profile=None):
    entry = json.loads((out / 'manifest.json').read_text())['workloads'][workload]
    source = out / 'inputs' / entry['folder']
    config = json.loads((source / 'original-config.json').read_text())
    for name, sha in entry['input_files_sha256'].items():
        if digest(source/name) != sha: raise ValueError('Frozen input changed: '+str(source/name))
    folder = out / 'replays' / entry['folder'] / policy
    folder.mkdir(parents=True)
    env = dict(os.environ, PYTHONHASHSEED='0', VLLM_USE_V1='0', HF_HUB_OFFLINE='1',
               VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               COLLECT_TRACE='1' if policy == 'NONE' else '0',
               UNICACHE_EXPECTED_TOKENS=str(source / 'expected-tokens.json'),
               UNICACHE_COMPLETIONS=str(folder / 'completed.json'))
    for key in ['UNICACHE_POLICY_CANDIDATE', 'UNICACHE_POLICY_INTERFACE', 'POLICY_AGENT_INTERFACE']:
        env.pop(key, None)
    for key in ['MPLCONFIGDIR', 'VLLM_CACHE_ROOT', 'VLLM_CONFIG_ROOT']:
        cache = folder / key; cache.mkdir(); env[key] = str(cache)
    cmd = [sys.executable, '-m', 'scripts.sim2real_replay', '--model', str(ROOT / 'models/Llama-3.2-1B-Instruct'),
           '--device', 'cpu', '--max-model-len', str(config['max_model_len']),
           '--max-num-seqs', str(config['max_num_seqs']), '--block-size', str(config['block_size']),
           '--simulator-enable-virtual-time', '--simulator-num-gpu-blocks', str(config['gpu_blocks']),
           '--simulator-num-cpu-blocks', str(config['cpu_blocks']), '--tensor_parallel_size', '1',
           '--eviction-policy', 'SimpleHYBRID' if policy == 'UniCache' else policy,
           '--simulator-ttft', config['ttft'], '--simulator-tpot', str(config['tpot']),
           '--simulator-output-dir', str(folder), '--simulator-task-config', str(source / 'replay.yaml'),
           '--log-level', 'WARNING']
    if policy != 'NONE': cmd.append('--enable-prefix-caching')
    if profile: cmd.extend(['--opt-block-use-map-path', str(profile)])
    write(folder / 'command.json', cmd)
    started = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    summaries = list(folder.glob('*/summary.json'))
    if len(summaries) != 1: raise ValueError('Expected one summary: ' + str(folder))
    parent = summaries[0].parent
    completed = json.loads((folder / 'completed.json').read_text())
    expected = json.loads((source / 'expected-tokens.json').read_text())
    if set(completed) != set(expected): raise ValueError('Incomplete replay')
    if policy == 'NONE':
        hit, prompt = 0, entry['prompt_tokens']
    else:
        with (parent / 'simulator_data.pkl').open('rb') as f: data = pickle.load(f)
        counts = data['request_id_2_hit_count_map'];mapping = data['request_id_2_conv_key_map']
        if len(counts) != len(expected) or {mapping[k] for k in counts} != set(expected):
            raise ValueError('Hit accounting does not cover every request')
        for request, (h, n) in counts.items():
            if n != len(expected[mapping[request]]['input_ids']) or not 0 <= h <= n:
                raise ValueError('Invalid hit/prompt accounting')
        hit = sum(v[0] for v in counts.values());prompt = sum(v[1] for v in counts.values())
        summary = json.loads(summaries[0].read_text())['hit_rate_summary']
        if hit != summary['global_hit_tokens_num'] or prompt != summary['global_tokens']:
            raise ValueError('Global/per-request counters disagree')
    result = {'workload': workload, 'policy': policy, 'requests': len(completed), 'hit_tokens': hit,
              'prompt_tokens': prompt, 'hit_ratio': hit / prompt, 'runtime_seconds': time.monotonic() - started,
              'summary': str(summaries[0]), 'completion_ledger': str(folder / 'completed.json'),
              'exact_prompt_and_output_tokens_verified': True}
    if policy == 'NONE':
        use_map = parent / 'use_map.json'
        if not use_map.exists(): raise ValueError('Missing offline OPT profile')
        result['opt_profile'] = str(use_map)
    result['artifacts_sha256'] = {str(p):digest(p) for p in
                                 [summaries[0],parent/'simulator_data.pkl',folder/'completed.json']}
    if policy == 'NONE':result['artifacts_sha256'][str(use_map)] = digest(use_map)
    write(folder / 'result.json', result)
    print(f'{workload}/{policy}: {100*result["hit_ratio"]:.4f}% ({len(completed)} verified requests)', flush=True)
    return result


def compare(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if digest(out/'gpu-chart-reference.csv') != manifest['gpu_reference_csv_sha256']:
        raise ValueError('GPU chart reference was modified')
    with (out / 'gpu-chart-reference.csv').open() as f:
        refs = list(csv.DictReader(f))
    rows = []
    for r in refs:
        path = out / 'replays' / FOLDERS[r['workload']] / r['policy'] / 'result.json'
        sim = json.loads(path.read_text()) if path.exists() else None
        if sim:
            for artifact,sha in sim['artifacts_sha256'].items():
                if digest(Path(artifact)) != sha: raise ValueError('Replay artifact changed: '+artifact)
        gpu = float(r['hit_ratio_percent'])
        difference = 100*sim['hit_ratio'] - gpu if sim and r['reference_kind'] == 'measured_chart' else None
        rows.append({'workload': r['workload'], 'policy': r['policy'], 'gpu_hit_ratio_percent': gpu,
                     'reference_kind': r['reference_kind'],
                     'sim_hit_ratio_percent': 100*sim['hit_ratio'] if sim else None,
                     'sim_minus_gpu_pp': difference, 'absolute_error_pp': abs(difference) if difference is not None else None,
                     'sim_verified_requests': sim['requests'] if sim else None})
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    measured = [r for r in rows if r['sim_minus_gpu_pp'] is not None]
    report = ['# Matched-subset simulation versus reported A100 results', '',
              'Llama-3.2-1B-Instruct; one reported A100 worker; TP=1; 512 blocks × 16 tokens; '
              'max context=8,192; max sequences=32; CPU cache=0. All frozen input bytes and timing parameters are preserved.', '',
              'GPU values are manually transcribed from the supplied image and rounded to 0.01 pp '
              '(rounding uncertainty about ±0.005 pp if rounded to nearest). Raw GPU completion logs were not provided. '
              'Seven ALFRED values labeled as estimates in that image are excluded from statistics. '
              'No GPU TTFT/ITL measurements are available in the image; this experiment evaluates hit-ratio transfer only. '
              'Simulator timing uses the supplied starter formula, without A100 recalibration.', '',
              f'Completed comparisons: {len(measured)}/53 measured chart cells; '
              f'{sum(r["sim_hit_ratio_percent"] is not None for r in rows)}/60 simulator policy cells.', '']
    for label, subset in [('All measured chart cells (including the series labeled OPT)', measured),
                          ('Online policies only', [r for r in measured if r['policy'] != 'OPT'])]:
        if subset:
            report.append(f'- {label}: approximate mean absolute error '
                          f'{sum(r["absolute_error_pp"] for r in subset)/len(subset):.3f} pp; '
                          f'maximum {max(r["absolute_error_pp"] for r in subset):.3f} pp.')
    for workload in FOLDERS:
        report += ['', f'## {workload}', '', '| Policy | GPU chart (%) | Simulator (%) | Sim − GPU (pp) | Reference |',
                   '|---|---:|---:|---:|---|']
        for r in [r for r in rows if r['workload'] == workload]:
            sim = f'{r["sim_hit_ratio_percent"]:.3f}' if r['sim_hit_ratio_percent'] is not None else '—'
            error = f'{r["sim_minus_gpu_pp"]:+.3f}' if r['sim_minus_gpu_pp'] is not None else '—'
            report.append(f'| {r["policy"]} | {r["gpu_hit_ratio_percent"]:.2f} | {sim} | {error} | {r["reference_kind"]} |')
    report += ['', '## Interpretation limits', '',
               '- Compare only measured rows; matching imputed values would not validate transfer.',
               '- The GPU OPT series is below online policies in several workloads. Its future-use profile/implementation is unverified; it is not treated as a proven upper bound.',
               '- All simulator replays must pass exact prompt/output token equality and complete request accounting before appearing in the table.',
               '- Content and arrival parameters match, but later turns are scheduled after request completion. Approximate CPU timing versus actual GPU timing can change request interleaving.',
               '- ALFRED has only two trajectories; CodeParrot has 100 requests but only 18 distinct full prompts. These subsets do not represent the full agent eval workloads.', '']
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    if measured:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(16, 15), squeeze=False)
        for ax, workload in zip(axes.flat, FOLDERS):
            cells = [r for r in rows if r['workload'] == workload]
            ys = list(range(len(cells)))
            ax.barh([v-.18 for v in ys], [r['gpu_hit_ratio_percent'] if r['reference_kind'] == 'measured_chart' else math.nan for r in cells],
                    height=.35, color='#4c78a8', label='GPU (chart, measured rows)')
            ax.barh([v+.18 for v in ys], [r['sim_hit_ratio_percent'] if r['sim_hit_ratio_percent'] is not None else math.nan for r in cells],
                    height=.35, color='#f28e2b', label='Simulator')
            ax.set_yticks(ys, [r['policy'] for r in cells]);ax.invert_yaxis()
            ax.set_title(workload);ax.set_xlabel('Token-weighted cache hit ratio (%)');ax.grid(axis='x', alpha=.2)
        axes.flat[-1].set_visible(False)
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower right', bbox_to_anchor=(.92,.15))
        fig.suptitle('Exact subset: CPU simulation versus reported A100 hit ratios', fontsize=16)
        fig.text(.5,.01,'512 blocks × 16 tokens. GPU reference is rounded chart data; estimated ALFRED bars are omitted.',ha='center')
        fig.tight_layout(rect=(0,.03,1,.96))
        for ext in ['png','pdf']:fig.savefig(out / ('comparison.'+ext),dpi=160)
        plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'result/sim2real-inputs/unicache-tested-traces-94149333d833')
    p.add_argument('--output', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--workloads', nargs='+', choices=list(FOLDERS), default=list(FOLDERS))
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    p.add_argument('--stage', choices=['prepare','run','compare'], default='run')
    args = p.parse_args()
    if args.jobs < 1:p.error('jobs must be positive')
    out = (args.output or ROOT/'result/sim2real-subset'/datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d_%H%M%S')).resolve()
    if args.stage == 'compare':
        compare(out);print('Report:',out/'report.md');return
    if not (out/'manifest.json').exists():
        out.mkdir(parents=True, exist_ok=False)
        prepare(args.bundle.resolve(),out,DEFAULT_IMAGE)
    else:
        manifest = json.loads((out/'manifest.json').read_text())
        for name, sha in manifest['code_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError('Code changed since preparation; use a fresh output folder: '+name)
    if args.stage == 'prepare':print('Prepared:',out);return
    errors=[]
    def run_workload(workload):
        profile=None
        if 'OPT' in args.policies:
            path=out/'replays'/FOLDERS[workload]/'NONE/result.json'
            result=json.loads(path.read_text()) if path.exists() else replay(out,workload,'NONE')
            profile=Path(result['opt_profile'])
        for policy in args.policies:
            path=out/'replays'/FOLDERS[workload]/policy/'result.json'
            if path.exists():continue
            try:replay(out,workload,policy,profile if policy=='OPT' else None)
            except Exception as e:
                errors.append({'workload':workload,'policy':policy,'error':str(e)})
                print('FAILED',workload,policy,str(e),flush=True)
    print('Results:',out,flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs={pool.submit(run_workload,w):w for w in args.workloads}
        for future in as_completed(jobs):
            try:future.result()
            except Exception as e:errors.append({'workload':jobs[future],'error':str(e)})
    compare(out)
    write(out/'completion.json',{'requested_workloads':args.workloads,'requested_policies':args.policies,'errors':errors,
                               'requested_runs_passed':not errors})
    print('Report:',out/'report.md',flush=True)
    if errors:raise SystemExit(1)


if __name__ == '__main__':
    main()
