import csv
import json
from pathlib import Path
import tempfile
import unittest
from cache_budget_sweep import geometry,tables

class SweepTests(unittest.TestCase):
    def test_model_memory_conversion(self):
        c=json.loads(Path('models/Llama-3.2-1B-Instruct/config.json').read_text())
        g=geometry(c,80)
        self.assertEqual(g['parameters'],1235814400)
        self.assertEqual(g['kv_block_bytes'],262144)
        self.assertEqual([int(f*g['available_kv_bytes']//g['kv_block_bytes']) for f in [.5,.7,.9]],[159125,222776,286426])
    def test_online_winner_selected_only_on_tune(self):
        cases={f'.5/x/{s}':{'budget':.5,'blocks':10,'trace':'x','split':s} for s in ['tune','eval']}
        # tables expects the full online baseline set before publishing headroom.
        from cache_budget_sweep import ONLINE
        results={}
        for s in ['tune','eval']:
            for p in ONLINE+['OPT']:
                v=.5
                if p=='LFU':v=.6 if s=='tune' else .4
                if p=='OPT':v=.8
                results[f'.5/x/{s}/{p}']={'hit_ratio':v}
        with tempfile.TemporaryDirectory() as path:
            tables(Path(path),results,cases)
            with (Path(path)/'headroom.csv').open() as f: rows=list(csv.DictReader(f))
            row=next(r for r in rows if r['split']=='eval')
            self.assertEqual(row['best_online_by_tune'],'LFU')
            self.assertAlmostEqual(float(row['online_minus_LRU_pp']),-10)
            self.assertAlmostEqual(float(row['OPT_minus_LRU_pp']),30)

if __name__=='__main__':unittest.main()
