import json
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import agent_budget_sweep as sweep


class BudgetAgentTests(unittest.TestCase):
    def test_report_selects_both_heuristics_on_tune_and_measures_them_on_eval(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);trace='sharegpt';blocks=16384
            model=out/'models/Llama-3.2-1B-Instruct/config.json'
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({'hidden_size':16,'num_attention_heads':2,
                                         'num_hidden_layers':2,'num_key_value_heads':1}))
            tune={'LRU':.8,'LFU':.7,'FIFO':.6,'UniCache':.9,'OPT':1.0}
            evaluation={'LRU':.4,'LFU':.95,'FIFO':.8,'UniCache':.3,'OPT':.99}
            for policy in sweep.runner.ALL_POLICIES:
                for split in ['tune','eval']:
                    p=sweep.cache_path(out,blocks,trace,policy,split)
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            sweep.write(out/'agent-runs'/str(blocks)/(trace+'.json'),{'run':str(out/'agent')})
            parsed={'table':{'Agent':{'tune':.91,'eval':.5}},'attempts':3}
            def score(_out,_blocks,_trace,policy,split):
                values=tune if split=='tune' else evaluation
                return {'token_weighted_hit_ratio':values.get(policy,.1)},out
            with patch.object(sweep,'ROOT',out),patch.object(sweep,'cached',side_effect=score), \
                 patch.object(sweep,'read_run'),patch.object(sweep,'validate',return_value=parsed):
                sweep.report(out,[blocks],[trace])
            with (out/'headroom.csv').open() as f:
                rows=list(csv.DictReader(f))
            for row in rows:
                self.assertEqual(row['best_simple_by_tune'],'LRU')
                self.assertEqual(row['best_online_by_tune'],'UniCache')
            row=next(r for r in rows if r['split']=='eval')
            self.assertAlmostEqual(float(row['best_simple_hit_ratio']),.4)
            self.assertAlmostEqual(float(row['Agent_minus_best_simple_pp']),10)
            self.assertAlmostEqual(float(row['best_online_hit_ratio']),.3)
            self.assertAlmostEqual(float(row['Agent_minus_best_online_pp']),20)
            self.assertAlmostEqual(float(row['OPT_minus_Agent_pp']),49)
            with (out/'agent-vs-opt-eval.csv').open() as f:
                opt_row=next(csv.DictReader(f))
            self.assertAlmostEqual(float(opt_row['OPT_minus_Agent_pp_remaining']),49)
            self.assertAlmostEqual(float(opt_row['headroom_attained_percent']),100*10/59)
            self.assertAlmostEqual(float(opt_row['lru_tune_hit_ratio']),.8)
            self.assertAlmostEqual(float(opt_row['agent_tune_hit_ratio']),.91)
            self.assertAlmostEqual(float(opt_row['Agent_minus_LRU_tune_pp']),11)
            self.assertAlmostEqual(float(opt_row['OPT_minus_LRU_tune_pp']),20)
            self.assertAlmostEqual(float(opt_row['OPT_minus_Agent_tune_pp']),9)
            with (out/'agent-vs-best-heuristic.csv').open() as f:
                heuristic_row=next(csv.DictReader(f))
            self.assertEqual(heuristic_row['best_heuristic_selected_on_tune'],'UniCache')
            self.assertAlmostEqual(float(heuristic_row['Best_minus_Agent_tune_pp']),-1)
            self.assertAlmostEqual(float(heuristic_row['Best_minus_Agent_eval_pp']),-20)
            self.assertAlmostEqual(float(heuristic_row['OPT_minus_LRU_tune_pp']),20)
            self.assertAlmostEqual(float(heuristic_row['OPT_minus_Agent_tune_pp']),9)
            self.assertIn('Best simple heuristic (tune)',(out/'report.md').read_text())

    def test_normal_cli_exit_before_submission_feedback_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'sandbox/feedback').mkdir(parents=True)
            stop=run/'stop.json';feedback=run/'sandbox/feedback/submission-0001.json'
            stop.write_text('{}');feedback.write_text('{}')
            os.utime(stop,(10,10));os.utime(feedback,(20,20))
            with self.assertRaisesRegex(ValueError,'agent_exited_while_submission_pending'):
                sweep.verify_feedback_barrier(run)
            os.utime(stop,(30,30))
            self.assertEqual(sweep.verify_feedback_barrier(run)['stop_after_final_feedback_seconds'],10)

    def test_persisted_console_excludes_agent_actions_and_multiline_text(self):
        target=io.StringIO();stream=sweep.ProgressOnly(target)
        stream.write('[Claude] Bash: secret-tool-details\nassistant continuation\n')
        stream.write('Submission 1: ');stream.write('ok\n')
        stream.write('HEARTBEAT trace: 1 minute\n')
        self.assertEqual(target.getvalue(),'Submission 1: ok\nHEARTBEAT trace: 1 minute\n')

    def test_cache_rejects_modified_private_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);parent=out/'replay';parent.mkdir()
            (parent/'summary.json').write_text('original')
            path=sweep.cache_path(out,16384,'sharegpt','LRU','tune')
            sweep.write(path,{'score':{'token_weighted_hit_ratio':.5},'parent':str(parent),
                             'sha256':{'summary.json':sweep.runner.digest(parent/'summary.json')}})
            self.assertEqual(sweep.cached(out,16384,'sharegpt','LRU','tune')[0]['token_weighted_hit_ratio'],.5)
            (parent/'summary.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                sweep.cached(out,16384,'sharegpt','LRU','tune')

    def test_candidate_always_executes_evaluator(self):
        original=sweep.runner.Evaluator.replay
        original_command=sweep.runner.claude_command
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'prompt.md').write_text('test')
            called=[]
            def real(self,*args,**kwargs):called.append(kwargs);return 'fresh'
            def main():
                obj=object.__new__(sweep.runner.Evaluator)
                self.assertEqual(obj.replay('LRU','tune',out/'candidate',candidate=out/'policy.py'),'fresh')
                self.assertEqual(obj.replay('LRU','tune',out/'baseline'),('cached',out))
                _,env=sweep.runner.claude_command(out,'claude:opus')
                self.assertEqual(env['CLAUDE_CODE_DISABLE_BACKGROUND_TASKS'],'1')
                self.assertEqual(env['BASH_DEFAULT_TIMEOUT_MS'],'1800000')
                self.assertEqual(env['BASH_MAX_TIMEOUT_MS'],'1800000')
            try:
                with patch.object(sweep.runner.Evaluator,'replay',real),patch.object(sweep.runner,'main',main),patch.object(sweep,'cached',return_value=('cached',out)) as cache:
                    sweep.run_one(out,16384,'sharegpt',30,'claude:opus')
                    self.assertEqual(cache.call_count,1)
                self.assertEqual(len(called),1)
            finally:
                sweep.runner.Evaluator.replay=original
                sweep.runner.claude_command=original_command

    def test_normal_auth_keeps_context_and_tool_restrictions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'prompt.md').write_text('test')
            with patch.dict(os.environ,{'UNICACHE_NORMAL_CLI_AUTH':'1','USER':'account'}),patch('shutil.which',return_value='/bin/claude'):
                command,env=sweep.runner.claude_command(p,'claude:opus')
                self.assertEqual(env['HOME'],os.environ['HOME'])
                self.assertEqual(env['USER'],'account')
                for flag in ['--safe-mode','--restricted','--no-session-persistence','--strict-mcp-config']:
                    self.assertIn(flag,command)
                self.assertEqual(command[command.index('--setting-sources')+1],'')
                self.assertEqual(command[command.index('--permission-mode')+1],'dontAsk')


if __name__=='__main__':unittest.main()
