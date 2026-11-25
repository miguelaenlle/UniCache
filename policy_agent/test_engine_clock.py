"""Policies must read the same clock as the core (virtual in sim, wall on GPU)."""
import time
import unittest

from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG
from vllm.simulator.timer_utils import GLOBAL_TIMER, engine_time


class EngineClockTest(unittest.TestCase):
    def setUp(self):
        self.saved = (GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time,
                      GLOBAL_TIMER.get_current_time())

    def tearDown(self):
        GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time = self.saved[0]
        GLOBAL_TIMER.set_time(self.saved[1])

    def test_virtual_time_in_simulation(self):
        GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time = True
        GLOBAL_TIMER.set_time(123.5)
        self.assertEqual(engine_time(), 123.5)

    def test_wall_time_on_gpu(self):
        GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time = False
        GLOBAL_TIMER.set_time(0.0)
        self.assertAlmostEqual(engine_time(), time.time(), delta=5)

    def test_linux_sandbox_has_no_unsafe_fallback(self):
        import os
        from pathlib import Path
        from policy_agent.bridge import PolicyFailure, sandbox_command
        if Path('/usr/bin/sandbox-exec').exists():
            self.skipTest('macOS uses Seatbelt')
        old = os.environ.pop('UNICACHE_POLICY_SANDBOX', None)
        try:
            with self.assertRaises(PolicyFailure):
                sandbox_command('')
        finally:
            if old is not None:
                os.environ['UNICACHE_POLICY_SANDBOX'] = old


if __name__ == '__main__':
    unittest.main()
