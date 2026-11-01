"""Transport ordering/isolation tests; no simulator or candidate source imports."""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

from policy_agent import events


class EventTransportTests(unittest.TestCase):
    def setUp(self):
        events._observers.clear()
        events._ordinal = 0

    def test_disabled_never_imports_simulator(self):
        with patch.dict(os.environ, {}, clear=True):
            events.emit('lookup', content_hash='opaque')
        self.assertEqual(events._ordinal, 0)

    def test_targeted_and_global_observations_keep_order(self):
        class Observer:
            def __init__(self, cache_id):
                self.cache_id = cache_id
                self.received = []
            def emit(self, event):
                self.received.append(event)
        gpu, cpu = Observer('gpu:0'), Observer('cpu:0')
        events.register(gpu)
        events.register(cpu)
        timer = types.ModuleType('vllm.simulator.timer_utils')
        timer.GLOBAL_TIMER = types.SimpleNamespace(get_current_time=lambda: 2.5)
        with patch.dict(os.environ, {'UNICACHE_POLICY_INTERFACE': 'v2'}), patch.dict(sys.modules, {'vllm.simulator.timer_utils': timer}):
            events.emit_to(gpu, 'block_allocated', block_id=42, generation=1)
            events.emit('sequence_step', block_ids=[42])
            events.emit_to(cpu, 'block_allocated', block_id=50, generation=1)
        self.assertEqual([e['event_id'] for e in gpu.received], [1, 2])
        self.assertEqual([e['event_id'] for e in cpu.received], [2, 3])
        self.assertEqual(gpu.received[0]['time'], 2.5)
        self.assertNotIn('cache_id', gpu.received[1])
        self.assertEqual(cpu.received[1]['cache_id'], 'cpu:0')


if __name__ == '__main__':
    unittest.main()
