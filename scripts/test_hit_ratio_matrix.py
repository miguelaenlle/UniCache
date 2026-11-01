"""Orchestration checks with synthetic outputs; no simulator runs or downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from hit_ratio_matrix import run_matrix, run_process


class MatrixTests(unittest.TestCase):
    def fake_replay(self, root, fail_profile=False):
        lock = threading.Lock()
        state = {'active': 0, 'peak': 0, 'finished_profiles': set()}

        def run(workload, policy, use_map):
            with lock:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                if policy == 'OPT':
                    self.assertIn(workload, state['finished_profiles'])
                    self.assertEqual(use_map, root / workload / 'NONE' / 'use_map.json')
            try:
                time.sleep(0.02)
                if policy == 'NONE' and fail_profile:
                    raise RuntimeError('synthetic profile failure')
                directory = root / workload / policy
                directory.mkdir(parents=True)
                summary = directory / 'summary.json'
                summary.write_text(json.dumps({'hit_rate_summary': {
                    'global_hit_rate': 0.5, 'total_requests': 2,
                    'global_hit_tokens_num': 8, 'global_tokens': 16},
                    'simulation_time': 1}))
                if policy == 'NONE':
                    (directory / 'use_map.json').write_text('{}')
                    with lock:
                        state['finished_profiles'].add(workload)
                return summary
            finally:
                with lock:
                    state['active'] -= 1
        return run, state

    def test_parallel_matches_serial_and_obeys_dependencies(self):
        outputs = []
        for jobs in [1, 3]:
            with tempfile.TemporaryDirectory() as directory:
                run, state = self.fake_replay(Path(directory))
                results = {}
                def record(workload, policy, result):
                    self.assertEqual(threading.current_thread(), threading.main_thread())
                    results[workload, policy] = {k: v for k, v in result.items() if k != 'summary'}
                run_matrix(['A', 'B'], ['LRU', 'OPT', 'SimpleHYBRID'], jobs,
                           run, record, threading.Event())
                self.assertEqual(state['peak'], jobs)
                self.assertEqual(len(results), 6)
                outputs.append(results)
        self.assertEqual(*outputs)

    def test_profile_failures_only_fail_opt(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.fake_replay(Path(directory), fail_profile=True)
            results = {}
            run_matrix(['A', 'B'], ['LRU', 'OPT'], 2, run,
                       lambda w, p, r: results.update({(w, p): r}), threading.Event())
            self.assertEqual(len(results), 4)
            for workload in ['A', 'B']:
                self.assertEqual(results[workload, 'LRU']['hit_ratio'], 0.5)
                self.assertIn('OPT profile failed', results[workload, 'OPT']['error'])

    def test_missing_profile_map_fails_opt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            results = {}
            run_matrix(['A'], ['OPT'], 2, lambda *args: path,
                       lambda w, p, r: results.update({p: r}), threading.Event())
            self.assertIn('use_map.json missing', results['OPT']['error'])

    def test_worker_failure_does_not_stop_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.fake_replay(Path(directory))
            def failing(workload, policy, use_map):
                if policy == 'LFU':
                    raise OSError('synthetic launch failure')
                return run(workload, policy, use_map)
            results = {}
            run_matrix(['A'], ['LFU', 'LRU'], 2, failing,
                       lambda w, p, r: results.update({p: r}), threading.Event())
            self.assertIn('error', results['LFU'])
            self.assertEqual(results['LRU']['hit_ratio'], 0.5)

    def test_cancellation_terminates_replay_process(self):
        cancelled = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pid'
            code = 'import os,time; print(os.getpid(),flush=True); time.sleep(60)'
            with path.open('w') as log, ThreadPoolExecutor(1) as pool:
                future = pool.submit(run_process, [sys.executable, '-c', code],
                                     dict(os.environ), log, cancelled)
                deadline = time.monotonic() + 5
                while not path.read_text() and time.monotonic() < deadline:
                    time.sleep(0.01)
                try:
                    pid = int(path.read_text().strip())
                finally:
                    cancelled.set()
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    future.result(timeout=7)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


if __name__ == '__main__':
    unittest.main()
