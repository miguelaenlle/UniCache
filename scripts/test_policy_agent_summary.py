import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from summarize_policy_agent import summarize

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def run_fixture(self,trace='sharegpt',name='run',blocks=50000):
        p=self.root/name;p.mkdir()
        policy=b'class Policy: pass\n'
        digest=hashlib.sha256(policy).hexdigest()
        rows=[{'submission':i,'status':'ok','policy_sha256':digest,
               'tune':{'token_weighted_hit_ratio':t},'eval':{'token_weighted_hit_ratio':e}}
              for i,t,e in [(0,.5,.8),(1,.6,.4),(2,.55,.9)]]
        data={'run.json':{'args':{'trace':trace,'serving_model':'model','gpu_blocks':blocks,'start_policy':'LRU','max_submissions':30}},
              'selection.json':{'submission':1},'stop.json':{'reason':'budget'},
              'baseline-results.json':{'LRU':{'tune':{'token_weighted_hit_ratio':.5},'eval':{'token_weighted_hit_ratio':.8}}}}
        for name,value in data.items():(p/name).write_text(json.dumps(value))
        (p/'policy.py').write_bytes(policy)
        (p/'submissions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (p/'comparison.csv').write_text('policy,kind,submission,tune_hit_ratio,eval_hit_ratio\nLRU,baseline,,0.5,0.8\nAgent,selected,1,0.6,0.4\n')
        return p
    def test_snapshot_replay_does_not_read_mutated_original(self):
        p=self.run_fixture();out=summarize([p],output=self.root/'out')
        (p/'comparison.csv').write_text('damaged')
        replay=summarize(manifest=out/'manifest.json',output=self.root/'replay')
        for name in ['eval.csv','tune.csv','summary.md','runs.csv']:
            self.assertEqual((out/name).read_bytes(),(replay/name).read_bytes())
        with (out/'eval.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(rows[-1]['ShareGPT'],'0.4') # tune winner, not eval winner
    def test_tampered_snapshot_rejected(self):
        out=summarize([self.run_fixture()],output=self.root/'out')
        (out/'inputs/sharegpt/policy.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'checksum'):
            summarize(manifest=out/'manifest.json',output=self.root/'replay')
    def test_duplicate_workload_and_mixed_budget_rejected(self):
        a=self.run_fixture();b=self.run_fixture(name='other')
        with self.assertRaisesRegex(ValueError,'Duplicate workload'):
            summarize([a,b],output=self.root/'out')
        c=self.run_fixture(trace='mashqa',name='third',blocks=10000)
        with self.assertRaisesRegex(ValueError,'mix'):
            summarize([a,c],output=self.root/'out')
    def test_inconsistent_selection_or_csv_rejected(self):
        a=self.run_fixture()
        (a/'selection.json').write_text('{"submission":2}')
        with self.assertRaisesRegex(ValueError,'best tune'):
            summarize([a],output=self.root/'out')
        (a/'selection.json').write_text('{"submission":1}')
        p=a/'comparison.csv';p.write_text(p.read_text().replace('0.6,0.4','nan,0.4'))
        with self.assertRaisesRegex(ValueError,'Invalid hit ratio'):
            summarize([a],output=self.root/'out')

if __name__=='__main__':unittest.main()
