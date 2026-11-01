"""CLI setup checks without contacting a coding model or reading credentials."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from policy_agent.runner import claude_command


class CliConfigTests(unittest.TestCase):
    def command(self, env):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder/'prompt.md').write_text('Optimize policy.py using tune feedback.')
            with patch.dict(os.environ, env, clear=True), patch('shutil.which', return_value='/bin/claude'):
                return claude_command(folder, 'claude:opus')

    def test_submission_commands_wait_for_feedback(self):
        cmd, env = self.command({})
        self.assertEqual(env['CLAUDE_CODE_DISABLE_BACKGROUND_TASKS'], '1')
        self.assertEqual(env['BASH_DEFAULT_TIMEOUT_MS'], '1800000')
        self.assertEqual(env['BASH_MAX_TIMEOUT_MS'], '1800000')
        self.assertIn('--restricted', cmd)
        self.assertIn('--safe-mode', cmd)
        self.assertIn('--no-session-persistence', cmd)
        self.assertNotIn('TaskOutput', cmd[cmd.index('--tools')+1])

    def test_normal_login_keeps_same_tool_restrictions(self):
        cmd, env = self.command({'UNICACHE_NORMAL_CLI_AUTH':'1', 'HOME':'/example/account',
                                  'USER':'test-user', 'UNRELATED_SECRET':'not-inherited'})
        self.assertEqual(env['HOME'], '/example/account')
        self.assertEqual(env['USER'], 'test-user')
        self.assertNotIn('UNRELATED_SECRET', env)
        self.assertIn('--restricted', cmd)
        self.assertIn('--safe-mode', cmd)
        self.assertEqual(cmd[cmd.index('--setting-sources')+1], '')
        self.assertEqual(cmd[cmd.index('--mcp-config')+1], '{"mcpServers":{}}')


if __name__ == '__main__':
    unittest.main()
