import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_experience_runs import collect, report


class CompareExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_fixture(self, name, agent, experience=False, prepare_only=False, date='2026-11-27',
                    baselines=None, tunes=(.5, .6)):
        folder = self.root/date/name
        folder.mkdir(parents=True)
        args = {'trace': 'sharegpt', 'serving_model': 'm', 'policy_interface': 'v1', 'gpu_blocks': 16384,
                'sessions': None, 'start_policy': 'LRU', 'max_submissions': 30,
                'prepare_only': prepare_only, 'experience': experience}
        (folder/'run.json').write_text(json.dumps({'args': args}))
        baselines = baselines or {'LRU': (.40, .50), 'LFU': (.45, .42), 'UniCache': (.55, .60), 'OPT': (.9, .9)}
        with (folder/'comparison.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['policy', 'kind', 'submission', 'tune_hit_ratio', 'eval_hit_ratio'])
            for policy, (tune, eval_) in baselines.items():
                w.writerow([policy, 'baseline', '', tune, eval_])
            w.writerow(['Agent', 'selected', 1, *agent])
        (folder/'submissions.jsonl').write_text(''.join(
            json.dumps({'submission': i, 'status': 'ok', 'tune': {'token_weighted_hit_ratio': t}})+'\n'
            for i, t in enumerate(tunes)))

    def test_deltas_use_lru_and_best_heuristic_chosen_on_tune(self):
        self.run_fixture('100000-sharegpt', (.60, .58))
        [run] = collect(self.root)
        self.assertEqual(run['best_heuristic'], 'UniCache')  # OPT never counts as a heuristic
        self.assertAlmostEqual(run['vs_lru_tune_pp'], 20)
        self.assertAlmostEqual(run['vs_lru_eval_pp'], 8)
        self.assertAlmostEqual(run['vs_best_tune_pp'], 5)
        self.assertAlmostEqual(run['vs_best_eval_pp'], -2)
        self.assertEqual(run['first_submission_beating_best_tune'], 1)

    def test_best_heuristic_is_tune_selected_even_if_another_wins_eval(self):
        self.run_fixture('100000-sharegpt', (.6, .6), baselines={'LRU': (.5, .7), 'FIFO': (.4, .9)})
        [run] = collect(self.root)
        self.assertEqual(run['best_heuristic'], 'LRU')
        self.assertAlmostEqual(run['vs_best_eval_pp'], -10)

    def test_arms_chains_and_filters(self):
        self.run_fixture('090000-sharegpt', (.5, .5), date='2026-11-20')
        self.run_fixture('100000-sharegpt', (.5, .5))
        self.run_fixture('110000-sharegpt', (.56, .55), experience=True)
        self.run_fixture('120000-sharegpt', (.58, .57), experience=True)
        self.run_fixture('130000-sharegpt', (.6, .6), prepare_only=True)
        runs = collect(self.root, since='2026-11-27')
        self.assertEqual([(r['arm'], r['chain']) for r in runs],
                         [('control', 1), ('experience', 1), ('experience', 2)])
        text = report(runs)
        self.assertIn('experience chain, eval vs best by position: -5.00 → -3.00', text)
        self.assertIn('* control:', text)

    def test_missing_baselines_leave_blank_deltas(self):
        self.run_fixture('100000-sharegpt', (.6, .6), baselines={'OPT': (.9, .9)}, tunes=(.6,))
        [run] = collect(self.root)
        self.assertIsNone(run['best_heuristic'])
        self.assertIsNone(run['vs_lru_tune_pp'])
        self.assertIn('| - |', report([run]))


if __name__ == '__main__':
    unittest.main()
