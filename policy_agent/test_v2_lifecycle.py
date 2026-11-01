"""Focused physical lifecycle tests, separate from end-to-end policy parity.

Run: .venv-sim312/bin/python -m unittest policy_agent.test_v2_lifecycle
The local RPC substitute executes the exact standalone source but removes subprocess
startup from these allocator tests; worker isolation is checked by the parity CLI.
"""
import os
import unittest
from unittest.mock import patch

from policy_agent import events
from policy_agent.v2_bridge import IndependentEvictor
from policy_agent.v2_shadow import NativeShadowEvictor
from policy_agent.v2_policies import source_for_policy
from vllm.core.block import prefix_caching_block as module
from vllm.core.evictor import EvictionPolicy, LRUEvictor, FIFOEvictor


class _LocalRPC:
    def __init__(self, source, config):
        scope = {}
        exec(source, scope)
        self.policy = scope['Policy'](config)
        self.initial = {'cache_mode': self.policy.CACHE_MODE}
        self.received = []

    def call(self, message):
        for event in message['events']:
            self.received.append(event)
            self.policy.on_event(event)
        return {'through_event_id': message['context']['through_event_id'],
                'victim': self.policy.evict(message['context']) if message['choose'] else None}

    def close(self):
        pass


class _NativeLRU(LRUEvictor):
    def __init__(self):
        super().__init__()
        self.decisions = []
    def evict(self):
        value = super().evict()
        self.decisions.append(value)
        return value


class _PassiveObserver:
    """Native algorithm plus raw-event recording; never changes its decisions."""
    def __init__(self):
        self.base = FIFOEvictor()
        self.config = {}
        self.cache_id = 'gpu:0'
        self.received = []
        events.register(self)
    def __getattr__(self, name):
        return getattr(self.base, name)
    def __contains__(self, block_id):
        return block_id in self.base
    def emit(self, event):
        self.received.append(event)
    def close(self):
        pass


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        events._observers.clear()
        events._ordinal = 0

    def tearDown(self):
        for observer in tuple(events._observers):
            observer.close()
        events._observers.clear()

    def exercise(self, v2, shadow=False, source_policy='LRU'):
        source = source_for_policy(source_policy)
        factory = (lambda _: IndependentEvictor(source)) if v2 else (lambda _: _NativeLRU())
        if shadow:
            factory = lambda _: NativeShadowEvictor(_NativeLRU(), source)
        env = {'UNICACHE_POLICY_INTERFACE': 'v2' if v2 and not shadow else 'v1',
               'POLICY_AGENT_INTERFACE': '', 'UNICACHE_POLICY_OBSERVE': '1' if shadow else '0'}
        with patch.dict(os.environ, env), patch.object(module, 'make_evictor', factory), patch('policy_agent.v2_bridge.V2Remote', _LocalRPC):
            allocator = module.PrefixCachingBlockAllocator(3, 8, eviction_policy=EvictionPolicy.LRU)
            trace = []

            def touch(block, now):
                ids = [block.block_id]
                if v2:
                    step = dict(sequence_id=int(now), block_ids=ids, now=now,
                                is_prefill=True, workload_class='ShareGPT-multi-turn_fixture',
                                output_tokens_len=0)
                    events.emit('sequence_step', **step)
                    events.emit('sequence_release', sequence_id=int(now), block_ids=ids,
                                last_step=step)
                allocator.mark_blocks_as_accessed(ids, now)

            a = allocator.allocate_immutable_block(None, list(range(8)))
            aid = a.block_id
            alias = allocator.allocate_immutable_block(None, list(range(8)))
            self.assertEqual(alias.block_id, aid)
            self.assertEqual(allocator._refcounter.get(aid), 2)
            self.assertNotIn(aid, allocator.evictor)
            allocator.free(alias)
            self.assertEqual(allocator._refcounter.get(aid), 1)
            self.assertNotIn(aid, allocator.evictor)

            mutable = allocator.allocate_mutable_block(None)
            mutable.append_token_ids([50, 51, 52])
            old_id = mutable.block_id
            fork = allocator.fork(mutable)[0]
            self.assertEqual(fork.block_id, old_id)
            fork.append_token_ids([53])
            self.assertNotEqual(fork.block_id, old_id)
            self.assertEqual(mutable.token_ids, [50, 51, 52])
            self.assertEqual(fork.token_ids, [50, 51, 52, 53])
            self.assertEqual(allocator.clear_copy_on_writes(), [(old_id, fork.block_id)])
            allocator.free(fork)
            allocator.free(mutable)
            allocator.mark_blocks_as_computed([])
            touch(a, 1.)
            allocator.free(a)
            self.assertIn(aid, allocator.evictor)

            for n in range(2, 8):
                block = allocator.allocate_immutable_block(None, [n] * 8)
                allocator.mark_blocks_as_computed([])
                touch(block, float(n))
                trace.append((block.block_id, block.content_hash))
                allocator.free(block)
            self.assertEqual(allocator.get_num_free_blocks(), 3)
            if v2:
                allocator.evictor.flush(False)
                received = list(allocator.evictor.remote.received)
                self.assertTrue(any(e['kind']=='refcount_changed' and e.get('reason')=='fork' and e['refcount']==2 for e in received))
                self.assertTrue(any(e['kind']=='block_activated' and e['refcount']==2 for e in received))
                self.assertTrue(any(e['kind']=='copy_on_write' for e in received))
                self.assertTrue(any(e['kind']=='block_allocated' and e['generation']>1 for e in received))
                decisions = [(e['block_id'], int(e['content_hash'])) for e in received if e['kind']=='block_evicted']
                old_observer = allocator.evictor
            else:
                decisions = list(allocator.evictor.decisions)
                old_observer = None
            self.assertGreater(len(decisions), 0)
            self.assertTrue(allocator.reset_prefix_cache())
            self.assertFalse(allocator._cached_blocks)
            if v2:
                self.assertNotIn(old_observer, events._observers)
                self.assertEqual(allocator.evictor.config['capacity'], 3)
                block = allocator.allocate_immutable_block(None, [999] * 8)
                self.assertEqual(allocator._refcounter.get(block.block_id), 1)
                touch(block, 9.)
                allocator.free(block)
                allocator.evictor.flush(False)
            return trace, decisions

    def test_passive_native_observation_preserves_fifo_tracker_and_decisions(self):
        results = []
        for observed in (False, True):
            env = {'UNICACHE_POLICY_INTERFACE': 'v1', 'POLICY_AGENT_INTERFACE': '',
                   'UNICACHE_POLICY_OBSERVE': '1' if observed else '0'}
            factory = (lambda _: _PassiveObserver()) if observed else (lambda _: FIFOEvictor())
            with patch.dict(os.environ, env), patch.object(module, 'make_evictor', factory):
                allocator = module.PrefixCachingBlockAllocator(2, 8, eviction_policy=EvictionPolicy.FIFO)
                self.assertEqual(allocator.eviction_policy, EvictionPolicy.FIFO)
                decisions = []
                for n in range(5):
                    block = allocator.allocate_immutable_block(None, [n] * 8)
                    allocator.mark_blocks_first_accessed_time([block.block_id], float(n))
                    decisions.append((block.block_id, block.content_hash))
                    allocator.free(block)
                results.append(decisions)
                if observed:
                    self.assertTrue(any(e['kind']=='lookup_completed' for e in allocator.evictor.received))
                    self.assertTrue(any(e['kind']=='block_allocated' and e['generation']>1 for e in allocator.evictor.received))
        self.assertEqual(*results)

    def test_shadow_rejects_a_different_strategy_at_first_victim_difference(self):
        from policy_agent.bridge import PolicyFailure
        with self.assertRaisesRegex(PolicyFailure, 'ShadowPolicyMismatch'):
            self.exercise(True, shadow=True, source_policy='LFU')

    def test_shared_active_cow_generation_reset_and_native_equivalence(self):
        native = self.exercise(False)
        independent = self.exercise(True)
        self.assertEqual(native, independent)
        shadow = self.exercise(True, shadow=True)
        self.assertEqual(native, shadow)


if __name__ == '__main__':
    unittest.main()
