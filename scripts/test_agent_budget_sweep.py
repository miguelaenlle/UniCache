import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import agent_budget_sweep as sweep


class BudgetAgentTests(unittest.TestCase):
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
