"""Run with simulator Python outside an enclosing sandbox: worker uses Seatbelt."""
import json
from pathlib import Path
import tempfile
import unittest

from policy_agent.bridge import PolicyFailure
from policy_agent.v2_bridge import IndependentEvictor, V2Remote

SOURCE = '''
class Policy:
    def __init__(self, config): self.blocks = {}
    def on_event(self, e):
        if e['kind'] == 'eligibility_changed':
            if e['eligible']: self.blocks[e['block_id']] = e['generation']
            else: self.blocks.pop(e['block_id'], None)
        if e['kind'] == 'block_evicted': self.blocks.pop(e['block_id'])
    def evict(self, context):
        b = min(self.blocks)
        return {'block_id': b, 'generation': self.blocks[b]}
'''


class WorkerTests(unittest.TestCase):
    def remote(self, source=SOURCE):
        r = V2Remote(source, {'capacity':8,'block_size':8,'api_version':2})
        self.addCleanup(r.close)
        return r

    def test_ordered_events_and_explicit_decision(self):
        r = self.remote()
        out = r.call({'events':[{'event_id':1,'kind':'eligibility_changed',
                                'eligible':True,'block_id':7,'generation':4}],
                      'choose':True,'context':{'through_event_id':1}})
        self.assertEqual(out, {'through_event_id':1,'victim':{'block_id':7,'generation':4}})

    def test_none_and_bool_victims_are_rejected(self):
        for expression in ('None', "{'block_id':True,'generation':1}"):
            r = self.remote(SOURCE[:SOURCE.index('    def evict')] +
                            f'    def evict(self, context): return {expression}\n')
            with self.assertRaises(PolicyFailure):
                r.call({'events':[],'choose':True,'context':{'through_event_id':0}})

    def test_event_order_and_ack_are_enforced(self):
        for message in ({'events':[{'event_id':0}], 'choose':False},
                        {'events':[], 'choose':True, 'context':{'through_event_id':5}}):
            with self.assertRaises(PolicyFailure):
                self.remote().call(message)

    def test_repo_future_data_and_network_are_denied(self):
        private = Path(__file__).resolve().parents[1]/'vllm/core/evictor.py'
        for expression in (f'open({str(private)!r}).read()',
                           '__import__("socket").create_connection(("127.0.0.1", 9))'):
            source = SOURCE[:SOURCE.index('    def evict')] + f'    def evict(self, context): return {expression}\n'
            with self.assertRaisesRegex(PolicyFailure, 'PermissionError|FileNotFoundError'):
                self.remote(source).call({'events':[],'choose':True,'context':{'through_event_id':0}})

    def test_numerical_dependencies_work_without_baseline_import(self):
        r = self.remote('import numpy, scipy.stats\n' + SOURCE)
        self.assertEqual(r.initial['cache_mode'],'prefix')

    def test_startup_mode_probe_does_not_construct_with_partial_context(self):
        source = SOURCE.replace('self.blocks = {}',
                                "self.blocks = {}; self.model = config['serving_model']")
        r = V2Remote(source, {}, mode_only=True)
        self.addCleanup(r.close)
        self.assertEqual(r.initial['cache_mode'], 'prefix')
        with self.assertRaisesRegex(PolicyFailure, 'Startup-only'):
            r.call({'events':[], 'choose':False})
        r = V2Remote(source, {'serving_model':'llama-3.2-1b-instruct'})
        self.addCleanup(r.close)
        self.assertEqual(r.initial['cache_mode'], 'prefix')

    def test_mode_cannot_change_in_constructor(self):
        source = SOURCE.replace('self.blocks = {}',
                                "self.blocks = {}; self.CACHE_MODE = 'disabled'")
        with self.assertRaisesRegex(PolicyFailure, 'class-level'):
            self.remote(source)

    def test_stale_victim_fails_host_validation(self):
        from unittest.mock import patch
        e = IndependentEvictor(SOURCE)
        self.addCleanup(e.close)
        e.eligible[3] = 'identity'
        e.generations[3] = 4
        with patch.object(e, 'flush', return_value={'block_id':3,'generation':2}):
            with self.assertRaisesRegex(PolicyFailure, 'stale'):
                e.evict()

    def test_host_rejects_malformed_victim_even_if_worker_validation_is_bypassed(self):
        from unittest.mock import patch
        e = IndependentEvictor(SOURCE)
        self.addCleanup(e.close)
        e.eligible[1] = 'identity'
        e.generations[1] = 1
        for victim in (None, {'block_id':True,'generation':1}, {'block_id':1},
                       {'block_id':1,'generation':1,'extra':'field'}):
            with self.subTest(victim=victim), patch.object(e, 'flush', return_value=victim):
                with self.assertRaisesRegex(PolicyFailure, 'integer'):
                    e.evict()
                self.assertIn(1,e.eligible)


if __name__ == '__main__':
    unittest.main()
