import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from policy_agent import runner


def hit(value):
    return {'token_weighted_hit_ratio': value}


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(runner, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(trace='sharegpt', serving_model='llama-3.2-1b-instruct',
                                    policy_interface='v1', gpu_blocks=16384, start_policy='LRU')
        self.manifest = {'source_session_limit': 1000}
        self.count = 0

    def run_folder(self, tunes, start_policy='LRU'):
        """Fake a finished run: one submission per tune score, each with an eval score."""
        self.count += 1
        out = self.root/'result/policy-agent/2026-11-27'/f'{self.count:06d}-sharegpt'
        sandbox = out/'sandbox'
        (sandbox/'experience').mkdir(parents=True)
        rows = []
        for i, tune in enumerate(tunes):
            folder = out/'internal'/f'submission-{i:04d}'
            folder.mkdir(parents=True)
            (folder/'policy.py').write_text(f'# run {self.count} submission {i}\n')
            rows.append({'submission': i, 'status': 'ok', 'tune': hit(tune), 'eval': hit(0.99)})
        rows.append({'submission': len(tunes), 'status': 'error', 'eval_error': 'x',
                     'error': f'Traceback in {out}/internal/worker.py'})
        (out/'stop.json').write_text(json.dumps({'reason': 'budget'}))
        evaluator = SimpleNamespace(
            out=out, rows=rows, args=SimpleNamespace(start_policy=start_policy),
            comparisons=[('LRU', {'tune': hit(.4), 'eval': hit(.5)}),
                         ('OPT', {'tune': hit(.9), 'eval': hit(.95)}),
                         ('LFU', {'eval': hit(.3)})])
        return sandbox, evaluator

    def store(self, **changes):
        return runner.experience_store(SimpleNamespace(**{**vars(self.args), **changes}), self.manifest)

    def test_store_key_ignores_start_policy_but_not_cache_size(self):
        self.assertEqual(self.store(), self.store(start_policy='UniCache'))
        self.assertNotEqual(self.store(), self.store(gpu_blocks=32768))
        self.assertEqual(self.store().name, 'sharegpt-llama-3.2-1b-instruct-v1-16384blocks-1000sessions')

    def test_load_creates_lessons_and_copies_store(self):
        sandbox = self.root/'sandbox'
        sandbox.mkdir()
        runner.load_experience(self.store(), sandbox)
        self.assertIn('No earlier runs', (sandbox/'experience/lessons.md').read_text())
        (self.store()/'lessons.md').write_text('# Lessons\nuse recency\n')
        other = self.root/'sandbox2'
        other.mkdir()
        runner.load_experience(self.store(), other)
        self.assertEqual((other/'experience/lessons.md').read_text(), '# Lessons\nuse recency\n')

    def test_save_keeps_eval_scores_and_paths_out_of_store(self):
        store = self.store()
        store.mkdir(parents=True)
        sandbox, evaluator = self.run_folder([.5, .6])
        runner.save_experience(store, sandbox, evaluator, [{'role': 'assistant', 'content': []}])
        for name in ['submissions.jsonl', 'runs.jsonl', 'best.json', 'conversation.jsonl']:
            text = (store/name).read_text()
            self.assertNotIn('"eval', text, name)  # no eval or eval_error keys
            self.assertNotIn('0.99', text, name)
            self.assertNotIn(str(evaluator.out), text, name)
        self.assertIn('<evaluator>', (store/'submissions.jsonl').read_text())
        run = json.loads((store/'runs.jsonl').read_text())
        self.assertEqual(run['baselines_tune_hit_ratio'], {'LRU': .4})  # OPT and tune-less baselines dropped
        self.assertEqual(run['best_submission'], 1)
        self.assertEqual(len((store/'submissions.jsonl').read_text().splitlines()), 3)

    def test_best_policy_only_replaced_by_higher_tune(self):
        store = self.store()
        store.mkdir(parents=True)
        for tunes, expected in [([.5, .6], 'run 1 submission 1'), ([.55], 'run 1 submission 1'),
                                ([.4, .7], 'run 3 submission 1')]:
            sandbox, evaluator = self.run_folder(tunes, start_policy='UniCache')
            runner.save_experience(store, sandbox, evaluator, [])
            self.assertIn(expected, (store/'best_policy.py').read_text())
        best = json.loads((store/'best.json').read_text())
        self.assertEqual((best['tune_hit_ratio'], best['start_policy']), (.7, 'UniCache'))
        self.assertEqual(len((store/'runs.jsonl').read_text().splitlines()), 3)

    def test_lessons_copied_back_but_not_through_symlink(self):
        store = self.store()
        store.mkdir(parents=True)
        sandbox, evaluator = self.run_folder([.5])
        (sandbox/'experience/lessons.md').write_text('# Lessons\nLFU loses on ShareGPT\n')
        runner.save_experience(store, sandbox, evaluator, [])
        self.assertIn('LFU loses', (store/'lessons.md').read_text())

        secret = self.root/'secret.txt'
        secret.write_text('host secret')
        sandbox, evaluator = self.run_folder([.5])
        (sandbox/'experience/lessons.md').symlink_to(secret)
        runner.save_experience(store, sandbox, evaluator, [])
        self.assertIn('LFU loses', (store/'lessons.md').read_text())

    def test_oversized_lessons_ignored(self):
        store = self.store()
        store.mkdir(parents=True)
        sandbox, evaluator = self.run_folder([.5])
        (sandbox/'experience/lessons.md').write_text('x' * (1024*1024 + 1))
        runner.save_experience(store, sandbox, evaluator, [])
        self.assertFalse((store/'lessons.md').exists())

    def test_prompt_mentions_experience_only_when_enabled(self):
        self.assertIn('experience/lessons.md', runner.EXPERIENCE)
        self.assertNotIn('experience/', runner.PROMPT)


if __name__ == '__main__':
    unittest.main()
