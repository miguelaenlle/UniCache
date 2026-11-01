"""Policy optimization runner; host evaluator, restricted Claude CLI and Seatbelt-isolated policy workers."""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from policy_agent.bridge import Remote

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'UniCache': 'SimpleHYBRID'}
POLICIES = ['UniCache', 'LRU', 'LFU', 'FIFO', 'LeCaR', 'ARC', 'WA', 'OPT']
ALL_POLICIES = POLICIES + ['OPT-Online', 'SimpleHYBRID-OPT', 'OFFSET-HEAD', 'OFFSET-TAIL', 'NONE']
TRACE_CONFIGS = {
    'sharegpt': 'TaskData/SharedGPT-multi-turn/tasks_config.yaml',
    'mashqa': 'TaskData/mashqa_data/tasks_config.yaml',
    'agentbank-apps': 'TaskData/AgentBankData/tasks_config.yaml',
    'alfred': 'TaskData/AgentBankData/tasks_config_alfred.yaml',
    'codeparrot': 'TaskData/Codeparrot/tasks_config.yaml',
}
PROMPT = '''Improve policy.py to increase cache hit ratio on the tune split.
Read policy_api.md and run_context.json. Use past and current cache information only.
Check your policy with: python smoke_test.py
You may write custom Python tests and run: python run_local.py tests/my_test.py
Local tests are offline, use no traces, and do not consume submission attempts.
Explain each change briefly in rationale.md, then run:
python submit.py --rationale-file rationale.md
Read the feedback and history.jsonl, then iterate. You get {budget} attempts;
accepted failures count. Eval results are hidden. Your best tune submission is kept.
Finish early: python submit.py --done --reason "<why you are stopping>"
'''
API = '''# Policy interface
Implement class Policy with __init__(), on_event(event), and evict(now).
on_event receives batches in their original order before evict is called.
- add: a block becomes eligible. Fields: block_id, content_hash,
  num_hashed_tokens, plus starting-policy metadata when available:
  last_accessed, access_count, task_type, is_decode.
- remove: block_id is no longer eligible; forget it.
- update: merge supplied fields for an existing block. Some policies emit an
  update without a block_id; ignore that event if not useful.
evict(now) returns an eligible integer block_id, or None to use the starting
policy's unchanged eviction decision. now is simulated seconds.
Only eviction decisions are editable in v1; allocator tracking stays unchanged.
Maintain your own state in Policy. No simulator imports, trace access or network.
Python standard library is available. Each callback batch has 30 seconds.
CPU limit: 600 seconds per policy worker. State resets for each replay.
stdout is reserved for the protocol. Exceptions fail the submission.
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def prepare(args):
    trace = getattr(args, 'trace', 'sharegpt')
    config_path = ROOT / TRACE_CONFIGS[trace]
    task = yaml.safe_load(config_path.read_text())['tasks'][0]
    source = (ROOT / task['data_path']).resolve()
    folder = source.parent
    count = args.sessions or task['max_record']
    if count < 2:
        raise ValueError('At least two sessions required')
    store = folder / 'agent-splits' / trace / f'v1-{count}'
    store.mkdir(parents=True, exist_ok=True)
    entries = []
    seen = set()
    with source.open() as f:
        for line in f:
            for sid, data in json.loads(line).items():
                if sid in seen or sid != data['id']:
                    raise ValueError('Duplicate or inconsistent session identity')
                seen.add(sid)
                if data['num_turns'] < 1:
                    raise ValueError('Empty session')
                for t in range(1, data['num_turns']+1):
                    turn = data[f'turn_{t}']
                    for field in ('input_ids', 'response_ids'):
                        if not turn[field] or any(type(x) is not int or x < 0 for x in turn[field]):
                            raise ValueError('Invalid token IDs')
                    if len(turn['input_ids']) + len(turn['response_ids']) > 131072:
                        raise ValueError('Unsupported context length; choose a supported source subset explicitly')
                entries.append((sid, data))
            if len(entries) >= count:
                break
    if len(entries) != count:
        raise ValueError('Requested session count is unavailable')
    rng = np.random.default_rng(task['seed']+task['task_id'])
    rng.shuffle(entries)
    arrivals = np.cumsum(np.random.default_rng(task['seed']+task['task_id']).exponential(
        1/task['lambda_rate'], count))
    configs, expected = {}, {}
    for split, lo, hi in [('tune', 0, count//2), ('eval', count//2, count)]:
        records = entries[lo:hi]
        datafile = store / f'{split}.jsonl'
        datafile.write_text(''.join(json.dumps({sid: data})+'\n' for sid, data in records))
        write(store / f'{split}-session-ids.json', [sid for sid, _ in records])
        frozen = {sid: float(arrivals[i]-arrivals[lo]) for i, (sid, _) in enumerate(entries) if lo <= i < hi}
        cfg = dict(task, data_path=str(datafile), max_record=len(records), frozen_arrivals=frozen)
        configs[split] = store / f'{split}.yaml'
        configs[split].write_text(yaml.safe_dump({'tasks': [cfg]}))
        expected[split] = {f'id_{sid}_turn_{t}': (len(data[f'turn_{t}']['input_ids']),
                                                        len(data[f'turn_{t}']['response_ids']))
                           for sid, data in records for t in range(1, data['num_turns']+1)}
    manifest = {'trace': trace, 'source': str(source), 'source_sha256': digest(source),
                'source_session_limit': count, 'chronology': 'simulated session starts; original generator shuffle/seed',
                'split': '50/50 by session start; whole sessions; independent cold caches',
                'configs': {s: {'path': str(p), 'sha256': digest(p),
                                'data_sha256': digest(store / f'{s}.jsonl')} for s, p in configs.items()}}
    write(store / 'trace.json', manifest)
    return configs, expected, manifest


class Evaluator:
    def __init__(self, args, out, configs, expected):
        self.args, self.out, self.configs, self.expected = args, out, configs, expected
        self.rows, self.comparisons = [], []
        self.closed = False

    def replay(self, policy, split, folder, candidate=None, use_map=None):
        print(f'Replay {policy}/{split}: {folder.name}', flush=True)
        folder.mkdir(parents=True, exist_ok=False)
        completed = folder / 'completed.json'
        env = dict(os.environ, VLLM_USE_V1='0', VLLM_NO_USAGE_STATS='1', PYTHONHASHSEED='0',
                   HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg',
                   COLLECT_TRACE='1' if policy == 'NONE' else '0', UNICACHE_COMPLETIONS=str(completed))
        env.pop('UNICACHE_POLICY_CANDIDATE', None)
        if candidate:
            env['UNICACHE_POLICY_CANDIDATE'] = str(candidate)
        for var in ['MPLCONFIGDIR', 'VLLM_CACHE_ROOT', 'VLLM_CONFIG_ROOT']:
            path = folder / var
            path.mkdir()
            env[var] = str(path)
        cmd = [sys.executable, '-m', 'policy_agent.replay', '--model', str(ROOT/'models/Llama-3.2-1B-Instruct'),
               '--device', 'cpu', '--max-num-seqs', '32', '--max-model-len', '131072', '--block-size', '8',
               '--simulator-enable-virtual-time', '--simulator-num-gpu-blocks', str(self.args.gpu_blocks),
               '--simulator-num-cpu-blocks', '50000', '--eviction-policy', ALIASES.get(policy, policy),
               '--simulator-ttft', '3.9e-05 * (batch_size ** 0.997) * (max_len ** 1.014)',
               '--simulator-tpot', '.01', '--simulator-output-dir', str(folder),
               '--simulator-task-config', str(self.configs[split]), '--log-level', 'WARNING']
        if policy != 'NONE':
            cmd.append('--enable-prefix-caching')
        if use_map:
            cmd += ['--opt-block-use-map-path', str(use_map)]
        write(folder/'command.json', cmd)
        with (folder/'run.log').open('w') as log:
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            except (subprocess.SubprocessError, OSError) as error:
                tail = (folder/'run.log').read_text()[-10000:]
                # Only policy frames and the final exception reach Claude. Full logs stay private.
                lines = tail.splitlines()
                public = [line for line in lines if line.startswith(('policy_agent.bridge.PolicyFailure:', 'PolicyFailure:'))]
                for index, line in enumerate(lines):
                    if 'File "policy.py"' in line:
                        public.extend(lines[index:index+2])
                if public and lines:
                    public.append(lines[-1])
                raise RuntimeError('\n'.join(public) or 'Replay failed; see private evaluator log') from error
        paths = list(folder.glob('**/summary.json'))
        if len(paths) != 1:
            raise RuntimeError('Expected exactly one simulator summary')
        parent = paths[0].parent
        with (parent/'simulator_data.pkl').open('rb') as f:
            metadata = pickle.load(f)  # Trusted simulator output, never candidate data.
        expected = self.expected[split]
        done = json.loads(completed.read_text())
        if done != {k: v[1] for k, v in expected.items()}:
            raise RuntimeError('Trace correctness failed: completed IDs/response lengths differ')
        actual_prompts = json.loads(completed.with_name('completed-prompts.json').read_text())
        if actual_prompts != {k:v[0] for k,v in expected.items()}:
            raise RuntimeError('Trace correctness failed: completed prompt lengths differ')
        if policy == 'NONE':
            # Upstream omits cache-hit records when prefix caching is disabled.
            # The independent completion ledger still verifies every request and token length.
            return {'token_weighted_hit_ratio':0., 'hit_tokens':0,
                    'prompt_tokens':sum(actual_prompts.values()), 'requests':len(done)}, parent
        counts = metadata['request_id_2_hit_count_map']
        mapping = metadata['request_id_2_conv_key_map']
        if {mapping[k] for k in counts} != set(expected) or len(counts) != len(expected):
            raise RuntimeError('Trace correctness failed: hit accounting omitted/duplicated requests')
        for key, (hit, prompt) in counts.items():
            if prompt != expected[mapping[key]][0] or not 0 <= hit <= prompt:
                raise RuntimeError('Trace correctness failed: token accounting mismatch')
        h = json.loads(paths[0].read_text())['hit_rate_summary']
        hit = sum(v[0] for v in counts.values())
        prompt = sum(v[1] for v in counts.values())
        if h['global_hit_tokens_num'] != hit or h['global_tokens'] != prompt or prompt <= 0:
            raise RuntimeError('Trace correctness failed: aggregate mismatch')
        return {'token_weighted_hit_ratio': hit/prompt, 'hit_tokens': hit,
                'prompt_tokens': prompt, 'requests': len(expected)}, parent

    def artifacts(self):
        valid = [r for r in self.rows if r['status'] == 'ok']
        best = max(valid, key=lambda r: (r['tune']['token_weighted_hit_ratio'], -r['submission'])) if valid else None
        if best:
            shutil.copyfile(self.out/'internal'/f"submission-{best['submission']:04d}"/'policy.py', self.out/'policy.py')
        with (self.out/'comparison.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['policy', 'kind', 'submission', 'tune_hit_ratio', 'eval_hit_ratio'])
            for name, scores in self.comparisons:
                w.writerow([name, 'baseline', '', *[scores.get(s, {}).get('token_weighted_hit_ratio', '') for s in ('tune', 'eval')]])
            if best:
                w.writerow(['Agent', 'selected', best['submission'], *[best.get(s, {}).get('token_weighted_hit_ratio', '') for s in ('tune', 'eval')]])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, split in zip(axes, ['tune', 'eval']):
            points = [r for r in self.rows if split in r]
            ax.plot([r['submission'] for r in points], [100*r[split]['token_weighted_hit_ratio'] for r in points], 'o-', label='Agent')
            for name, scores in self.comparisons:
                if split in scores:
                    ax.axhline(100*scores[split]['token_weighted_hit_ratio'], linestyle='--', alpha=.6, label=name)
            if best and split in best:
                ax.plot(best['submission'], 100*best[split]['token_weighted_hit_ratio'], '*', markersize=14, color='black', label='Selected by tune')
            ax.set(title=split, xlabel='Submission', ylabel='Token-weighted hit ratio (%)')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].legend(fontsize=7, ncol=2 if len(self.comparisons)>8 else 1)
        fig.tight_layout()
        fig.savefig(self.out/'hit-ratio.png')
        plt.close(fig)
        (self.out/'submissions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        write(self.out/'selection.json', {'submission': best['submission'] if best else None})

    def submit(self, source, rationale, number):
        print(f'Submission {number}/{self.args.max_submissions}: evaluating tune', flush=True)
        folder = self.out/'internal'/f'submission-{number:04d}'
        folder.mkdir()
        candidate = folder/'policy.py'
        candidate.write_text(source)
        (folder/'rationale.md').write_text(rationale)
        row = {'submission': number, 'policy_sha256': digest(candidate), 'rationale': rationale,
               'attempts_used': number, 'attempts_remaining': self.args.max_submissions-number}
        try:
            row['tune'], _ = self.replay(self.args.start_policy, 'tune', folder/'tune', candidate)
            row['status'] = 'ok'
        except Exception as error:
            row.update(status='validity_failed' if any(x in str(error) for x in ('PolicyTimeout', 'PermissionError')) else 'correctness_failed', error=str(error))
        if row['status'] == 'ok':
            try:
                print(f'Submission {number}: tune hit ratio {row["tune"]["token_weighted_hit_ratio"]:.2%}; evaluating eval', flush=True)
                row['eval'], _ = self.replay(self.args.start_policy, 'eval', folder/'eval', candidate)
            except Exception as error:
                row['eval_error'] = str(error)
        self.rows.append(row)
        public = {k:v for k,v in row.items() if k not in ('eval', 'eval_error')}
        # Remove host paths from errors. Raw traceback remains in internal logs.
        if 'error' in public:
            public['error'] = public['error'].replace(str(self.out), '<evaluator>')
        sandbox = self.out/'sandbox'
        with (sandbox/'history.jsonl').open('a') as f:
            f.write(json.dumps(public)+'\n')
        write(sandbox/'feedback'/f'submission-{number:04d}.json', public)
        self.artifacts()
        print(f'Submission {number}: {row["status"]}; {row["attempts_remaining"]} attempts left', flush=True)
        return public


def claude_command(sandbox, coding_model):
    claude = shutil.which('claude')
    if not claude:
        raise RuntimeError('Claude CLI not installed')
    env = {'PATH':f'{sandbox / "bin"}:/usr/bin:/bin:/opt/homebrew/bin',
           'HOME':str(sandbox/'home'),'TMPDIR':str(sandbox/'tmp'), 'PYTHONNOUSERSITE':'1'}
    for key in ['ANTHROPIC_API_KEY','CLAUDE_CODE_OAUTH_TOKEN']:
        if os.environ.get(key):
            env[key] = os.environ[key]
    cmd = [claude,'--print','--output-format','stream-json','--verbose','--model',coding_model.split(':',1)[1],
           '--restricted','--safe-mode','--no-session-persistence','--debug-file','/dev/null','--permission-mode','dontAsk','--permission-prompts','none','--setting-sources','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
           '--tools','Read,Edit,Write,Bash','--allowedTools','Read,Edit,Write,Bash(python submit.py *),Bash(python smoke_test.py),Bash(python run_local.py *)',
           '--append-system-prompt',(sandbox/'prompt.md').read_text(),
           'Read policy_api.md, run_context.json and policy.py. Follow the task prompt.']
    return cmd, env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['run'])
    p.add_argument('--trace', choices=list(TRACE_CONFIGS), required=True)
    p.add_argument('--serving_model', choices=['llama-3.2-1b-instruct'], required=True)
    p.add_argument('--coding_model', default='claude:opus')
    p.add_argument('--start-policy', choices=[x for x in POLICIES if x != 'OPT'], default='UniCache')
    p.add_argument('--compare', default=','.join(POLICIES), help='Comma-separated policies, all, or empty string')
    p.add_argument('--require-baselines', nargs='+', choices=ALL_POLICIES, default=[], help='Do not start Claude unless these baselines pass both splits')
    p.add_argument('--max-submissions', type=int, default=30)
    p.add_argument('--debug', action='store_true', help='Save the full Claude event stream to claude.log')
    p.add_argument('--prepare-only', action='store_true', help='Build sandbox and splits without simulations or Claude')
    p.add_argument('--sessions', type=int, help='Explicit source subset override; default authors standalone limit (1000)')
    p.add_argument('--gpu-blocks', type=int, default=50000)
    args = p.parse_args()
    if not args.coding_model.startswith('claude:') or not args.coding_model.split(':',1)[1]:
        p.error('Only claude:<model> is supported')
    if args.max_submissions < 1 or args.gpu_blocks < 1:
        p.error('Budget and cache capacity must be positive')
    comparisons = list(ALL_POLICIES) if args.compare == 'all' else (args.compare.split(',') if args.compare else [])
    if any(x not in ALL_POLICIES for x in comparisons):
        p.error('Unsupported comparison policy')
    if any(x not in comparisons and x != args.start_policy for x in args.require_baselines):
        p.error('Required baselines must also be selected for comparison')
    if sys.version_info < (3,12):
        p.error('Use the Python 3.12 simulator environment')
    if not Path('/usr/bin/sandbox-exec').exists():
        p.error('V1 currently requires macOS Seatbelt; no unsafe fallback')
    if not args.prepare_only and not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('CLAUDE_CODE_OAUTH_TOKEN')):
        p.error('Export ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN for the isolated Claude CLI, or use --prepare-only')
    configs, expected, manifest = prepare(args)
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d/%H%M%S-%f')
    out = ROOT/'result/policy-agent'/(stamp + '-' + args.trace)
    sandbox = out/'sandbox'
    for name in ['feedback', 'requests', 'home', 'tmp', 'bin', 'tests', 'local-output']:
        (sandbox/name).mkdir(parents=True, exist_ok=True)
    (sandbox/'bin/python').symlink_to(Path(sys.executable).resolve())
    (out/'internal').mkdir()
    for name in ['submit.py', 'local_client.py', 'run_local.py', 'smoke_test.py']:
        shutil.copyfile(ROOT/'policy_agent'/name, sandbox/name)
    shutil.copyfile(ROOT/'policy_agent/starter.py', sandbox/'policy.py')
    (sandbox/'rationale.md').write_text('Keep the starting policy unchanged.\n')
    (sandbox/'prompt.md').write_text(PROMPT.format(budget=args.max_submissions))
    (sandbox/'policy_api.md').write_text(API)
    (sandbox/'history.jsonl').touch()
    write(sandbox/'run_context.json', {'trace':args.trace, 'serving_model':args.serving_model,
          'starting_policy':args.start_policy, 'max_submissions':args.max_submissions,
          'block_size':8, 'gpu_blocks':args.gpu_blocks, 'cpu_blocks':50000,
          'split':'50/50 by simulated session start, whole sessions, cold caches',
          'tune_requests':len(expected['tune']), 'source_session_limit':manifest['source_session_limit']})
    write(out/'run.json', {'args':vars(args), 'trace':manifest,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'python':sys.version, 'model_checksums':{str(x.relative_to(ROOT)):digest(x) for x in (ROOT/'models/Llama-3.2-1B-Instruct').glob('*.json')},
          'limitations':['Eviction controller retains starting-policy tracking hooks; None delegates unchanged.',
                        'Claude uses restricted tool permissions, not OS isolation; web tools disabled and Bash limited to submission and offline local-test clients. API network remains available.',
                        'Worker has CPU and callback deadlines; strict resident-memory cap not implemented.',
                        'Original virtual-time gap generation is completion-relative, not a fixed wall-clock replay.']})
    print(f'Results: {out}', flush=True)
    if args.prepare_only:
        print('Prepared only; no simulations or Claude calls made.')
        return
    # Fail closed if OS isolation cannot be activated, before spending on Claude.
    probe = Remote((ROOT/'policy_agent/starter.py').read_text())
    probe.close()
    evaluator = Evaluator(args,out,configs,expected)
    profiles = {}
    for policy in dict.fromkeys([args.start_policy, *comparisons]):
        scores = {}
        for split in ['tune','eval']:
            print(f'Baseline {policy}/{split}', flush=True)
            basefolder = out/'internal'/'baselines'/policy/split
            use_map = None
            try:
                if policy == 'OPT':
                    profile_score, parent = evaluator.replay('NONE',split,basefolder/'profile')
                    profiles[split] = profile_score
                    use_map = parent/'use_map.json'
                if policy == 'NONE' and split in profiles:
                    scores[split] = profiles[split]
                else:
                    scores[split], _ = evaluator.replay(policy,split,basefolder/'replay',use_map=use_map)
            except Exception as error:
                if policy == args.start_policy:
                    raise
                scores[split + '_error'] = str(error)
        write(out/'baseline-results.json', dict(evaluator.comparisons + [(policy,scores)]))
        evaluator.comparisons.append((policy,scores))
    for required in args.require_baselines:
        if not all(split in dict(evaluator.comparisons)[required] for split in ('tune', 'eval')):
            write(out/'stop.json', {'reason':'required_baseline_failed','policy':required})
            evaluator.artifacts()
            raise RuntimeError(f'Required baseline {required} failed; refusing to start Claude. See baseline-results.json.')
    zero = evaluator.submit((sandbox/'policy.py').read_text(), 'Unchanged starting policy',0)
    baseline = dict(evaluator.comparisons)[args.start_policy]
    if any(evaluator.rows[0].get(s) != baseline[s] for s in ['tune','eval']):
        raise RuntimeError('Starting-controller parity failed; refusing to launch Claude')
    stop = threading.Event()
    def serve():
        while not stop.is_set():
            for request in sorted((sandbox/'requests').glob('*.json')):
                response = sandbox/'feedback'/request.name
                if response.exists():
                    continue
                try:
                    if request.stat().st_size > 1024*1024:
                        raise ValueError('Request exceeds 1 MiB')
                    if request.is_symlink():
                        raise ValueError('Symlink requests are not permitted')
                    message = json.loads(request.read_text())
                    if message.get('action') in ('local_test','smoke_test'):
                        from policy_agent.local_runner import run_test
                        print(f'Local test: {message.get("script", "built-in smoke test")}', flush=True)
                        write(response, run_test(sandbox,message))
                        continue
                    n = len(evaluator.rows)
                    if evaluator.closed or n > args.max_submissions:
                        raise ValueError('Submission budget exhausted or run closed')
                    if message.get('done'):
                        evaluator.closed = True
                        write(out/'stop.json', {'reason': str(message.get('reason','done'))})
                        write(response, {'ok':True,'output':'Run completed. Best tune submission selected.'})
                        continue
                    if not isinstance(message.get('source'),str) or not str(message.get('rationale','')).strip():
                        raise ValueError('Source and rationale required')
                    row = evaluator.submit(message['source'],message['rationale'],n)
                    text = f"Submission {n}: {row['status'].replace('_',' ').upper()}\n"
                    if 'tune' in row:
                        h = row['tune']
                        text += f"Tune hit ratio: {h['token_weighted_hit_ratio']:.2%}\nHit tokens: {h['hit_tokens']:,} / {h['prompt_tokens']:,}\nRequests: {h['requests']:,}\n"
                    if 'error' in row:
                        text += '\n'+row['error']+'\n'
                    text += f"Attempts left: {row['attempts_remaining']}\nFeedback: feedback/submission-{n:04d}.json"
                    write(response, {'ok':row['status']=='ok','output':text})
                    if n == args.max_submissions:
                        evaluator.closed = True
                except Exception as error:
                    write(response, {'ok':False,'output':str(error)})
            time.sleep(.2)
    thread = threading.Thread(target=serve,daemon=True)
    thread.start()
    cmd, env = claude_command(sandbox, args.coding_model)
    try:
        with ((out/'claude.log').open('w') if args.debug else nullcontext(None)) as log:
            print(f'Starting {args.coding_model}; streaming activity to terminal' + (' and claude.log' if args.debug else ' (agent log disabled)'), flush=True)
            # Stream CLI events through the host, keeping logs outside Claude's working directory.
            process = subprocess.Popen(cmd,cwd=sandbox,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,text=True,bufsize=1)
            first_output = threading.Event()
            def stream():
                for line in process.stdout:
                    first_output.set()
                    if log is not None:
                        log.write(line)
                        log.flush()
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'assistant':
                            for item in event.get('message',{}).get('content',[]):
                                if item.get('type') == 'text':
                                    print('[Claude] ' + item['text'], flush=True)
                                elif item.get('type') == 'tool_use':
                                    print('[Claude] ' + item.get('name','tool') + ': ' + str(item.get('input',{}))[:350], flush=True)
                        elif event.get('type') == 'result':
                            print('[Claude] ' + str(event.get('result',event.get('errors',event.get('subtype','finished')))), flush=True)
                        elif event.get('type') == 'system':
                            print('[Claude] ' + event.get('subtype','status'), flush=True)
                    except ValueError:
                        print('[Claude] ' + line.rstrip(), flush=True)
            reader = threading.Thread(target=stream,daemon=True)
            reader.start()
            closed_at = None
            started_at = time.monotonic()
            while process.poll() is None:
                if not first_output.is_set() and time.monotonic()-started_at > 90:
                    print('Claude emitted no startup output within 90 seconds; stopping. Use --debug for full agent logging.', flush=True)
                    write(out/'stop.json', {'reason':'claude_startup_timeout'})
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                if evaluator.closed:
                    closed_at = closed_at or time.monotonic()
                    if time.monotonic()-closed_at > 10:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                time.sleep(.5)
            status = process.wait()
            reader.join(timeout=5)
        if not (out/'stop.json').exists():
            write(out/'stop.json', {'reason':'budget' if evaluator.closed else ('agent_exit' if status==0 else 'agent_error'), 'exit_code':status})
    finally:
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        stop.set()
        thread.join()
        evaluator.artifacts()
    print(f'Finished. Results: {out}')


if __name__ == '__main__':
    main()
