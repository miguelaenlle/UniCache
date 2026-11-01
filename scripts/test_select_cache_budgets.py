import json
from pathlib import Path
import tempfile
import unittest
from select_cache_budgets import select

class SelectionTests(unittest.TestCase):
    def fixture(self,root,scores,splits=None,passed=True):
        p=root/'pilot';p.mkdir()
        (p/'completion.json').write_text(json.dumps({'passed':passed}))
        (p/'manifest.json').write_text(json.dumps({'args':{'splits':splits or ['tune']},'traces':{'x':{}},'budgets':{str(b):b for b in scores}}))
        results={}
        for b,ratios in scores.items():
            for policy,ratio in zip(['LRU','SimpleHYBRID','OPT'],ratios):results[f'{b}blocks/x/tune/{policy}']={'hit_ratio':ratio}
        (p/'results.json').write_text(json.dumps(results))
        return p
    def test_demonstrated_online_gain_beats_larger_oracle_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root,{10:(.2,.3,.9),20:(.4,.6,.8)})
            out=select(p,root/'selection.json')
            self.assertEqual(out['selections']['x']['selected']['blocks'],20)
            self.assertTrue((root/'selection.sh').exists())
    def test_no_online_gain_falls_back_to_oracle_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root,{10:(.2,.19,.9),20:(.4,.4,.8)})
            out=select(p,root/'selection.json')
            self.assertEqual(out['selections']['x']['selected']['blocks'],10)
            self.assertIn('oracle-only',out['selections']['x']['evidence'])
    def test_eval_selection_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root,{10:(.2,.3,.9)},splits=['tune','eval'])
            with self.assertRaisesRegex(ValueError,'tune-only'):select(p,root/'selection.json')
    def test_failed_pilot_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root,{10:(.2,.3,.9)},passed=False)
            with self.assertRaisesRegex(ValueError,'successful'):select(p,root/'selection.json')

if __name__=='__main__':unittest.main()
