"""Synthetic branch/numerical checks for editable v2 reference policies.

No trace data or simulator imports. Runtime code is unchanged. Every lifecycle,
access, miss and accounting observation is provided through the public event API;
private state is inspected only by this validation suite.
"""
import math
import unittest
from unittest.mock import patch

import numpy as np
from policy_agent.v2_policies import source_for_policy


def make(name, capacity=16, **config):
    scope = {}
    exec(source_for_policy(name), scope)
    policy = scope['Policy']({'capacity': capacity, 'gpu_blocks': capacity,
                              'block_size': 8, **config})
    return policy, scope


def event(policy, kind, time=1., **fields):
    policy.on_event({'kind': kind, 'time': time, **fields})


def create(policy, block_id, generation=1):
    event(policy, 'block_allocated', block_id=block_id, generation=generation)
    event(policy, 'hash_binding', block_id=block_id,
          content_hash=str(block_id), depth=8 * (block_id + 1))


def access(policy, block_id, now, task='ShareGPT-multi-turn_validation', prefill=True):
    seq = int(now * 1000) + block_id
    step = dict(sequence_id=seq, block_ids=[block_id], now=now, is_prefill=prefill,
                workload_class=task, prompt_tokens_len=8, output_tokens_len=0)
    event(policy, 'sequence_step', time=now, **step)
    event(policy, 'sequence_release', time=now, sequence_id=seq,
          block_ids=[block_id], last_step=step)


def eligible(policy, block_id):
    event(policy, 'eligibility_changed', block_id=block_id, eligible=True,
          content_hash=str(block_id), depth=8 * (block_id + 1))


def account(policy, ordinal, category, hit_tokens):
    event(policy, 'prefill_accounted', time=float(ordinal),
          request_id=f'r{ordinal}', accounting_ordinal=ordinal,
          task_category=category, cached_prompt_tokens=hit_tokens,
          total_prompt_tokens=max(1024, hit_tokens))


def expected_penalty(weights, expert, discount, elapsed):
    # Independent exponential-weights equation; float32 is the reference contract.
    reward = np.zeros(2, dtype=np.float32)
    reward[expert] = -(discount ** elapsed)
    value = weights * np.exp(np.float32(.45) * reward)
    value /= value.sum()
    if value[0] >= .99:
        return np.array([.99, .01], dtype=np.float32)
    if value[1] >= .99:
        return np.array([.01, .99], dtype=np.float32)
    return value


class V2NumericalBranches(unittest.TestCase):
    def test_wa_prefill_samples_decode_exclusion_and_second_fit(self):
        policy, scope = make('WA', wa_fit_interval=12)
        create(policy, 0)
        for now in (0., 2., 4., 6., 8., 10.):
            access(policy, 0, now)
        helper = scope['GLOBAL_WA_HELPER']
        task = scope['TaskType'].MULTI_TURN_TEXT
        self.assertEqual(helper.reuse_time[task], [2.] * 5)
        self.assertAlmostEqual(helper.get_reuse_probability(task, 3.), math.exp(-1.5))
        self.assertAlmostEqual(helper.fitted_params[task][0], .5)
        self.assertEqual(helper.last_fit_time, 10.)
        self.assertEqual(helper.reuse_time[task], [])
        access(policy, 0, 11., prefill=False)
        self.assertEqual(helper.all_data[task], [2.] * 5)
        for now in (14., 18.):
            access(policy, 0, now)
        event(policy, 'engine_iteration', time=19., prefill_counter=0)
        self.assertAlmostEqual(helper.get_reuse_probability(task, 3.), math.exp(-1.5))
        self.assertEqual(helper.last_fit_time, 10.)
        for now in (22., 26., 30.):
            access(policy, 0, now)
        self.assertEqual(helper.reuse_time[task], [4.] * 5)
        self.assertAlmostEqual(helper.get_reuse_probability(task, 3.), math.exp(-.75))
        self.assertAlmostEqual(helper.fitted_params[task][0], .25)
        self.assertEqual(helper.last_fit_time, 30.)
        self.assertEqual(helper.reuse_time[task], [])
        self.assertEqual(helper.all_data[task], [2.] * 5 + [4.] * 5)

    def test_unicache_window_rollover_and_inactive_category_decay(self):
        policy, scope = make('UniCache')
        for ordinal in range(1, 502):
            account(policy, ordinal, 'text', ordinal)
        stats = scope['_OUTPUT'].tracked_hit_stats
        self.assertEqual(len(stats['text']), 500)
        self.assertEqual(list(stats['text']), list(range(2, 502)))
        account(policy, 502, 'agentic', 10)
        self.assertEqual(len(stats['text']), 500)
        self.assertAlmostEqual(stats['text'][0], 2 * .95)
        self.assertAlmostEqual(stats['text'][-1], 501 * .95)
        account(policy, 503, 'text', 100)
        self.assertEqual(len(stats['text']), 500)
        self.assertAlmostEqual(stats['text'][0], 3 * .95)
        self.assertEqual(stats['text'][-1], 100)
        self.assertAlmostEqual(stats['agentic'][0], 9.5)
        self.assertEqual(scope['_OUTPUT'].step_track, 503)

    def test_unicache_alpha_update_exactly_once_at_tenth_accounting_boundary(self):
        policy, _ = make('UniCache')
        for block_id in range(12):
            create(policy, block_id)
            access(policy, block_id, float(block_id + 1),
                   task='alfred_validation' if block_id % 2 else 'ShareGPT-multi-turn_validation')
            eligible(policy, block_id)
        for ordinal in range(1, 10):
            account(policy, ordinal, 'agentic' if ordinal % 2 else 'text', 100)
        algorithm = policy.algorithm
        with patch.object(algorithm, 'update_alpha_balance', wraps=algorithm.update_alpha_balance) as update:
            policy.evict({'now': 100.})
            self.assertEqual(update.call_count, 0)
            account(policy, 10, 'text', 100)
            policy.evict({'now': 101.})
            self.assertEqual(update.call_count, 1)
            self.assertEqual(algorithm._last_alpha_update_step, 10)
            policy.evict({'now': 102.})
            self.assertEqual(update.call_count, 1)
            account(policy, 11, 'agentic', 100)
            policy.evict({'now': 103.})
            self.assertEqual(update.call_count, 1)
            for ordinal in range(12, 21):
                account(policy, ordinal, 'text', 100)
            policy.evict({'now': 104.})
            self.assertEqual(update.call_count, 2)
            self.assertEqual(algorithm._last_alpha_update_step, 20)

    def test_unicache_alpha_matches_efficiency_temperature_smoothing_equation(self):
        policy, scope = make('UniCache')
        for block_id in range(4):
            create(policy, block_id)
            access(policy, block_id, float(block_id + 1),
                   task='alfred_validation' if block_id % 2 else 'ShareGPT-multi-turn_validation')
            eligible(policy, block_id)
        account(policy, 1, 'text', 100)
        account(policy, 2, 'agentic', 25)
        # Two eligible blocks per category. Accounting decays text to 95 first.
        eps = .001
        hits = np.array([95., 25.])
        occupancy = 2 / (4 + eps)
        efficiency = (hits / (hits.sum() + eps)) / (occupancy + eps)
        scaled = (efficiency / (efficiency.mean() + eps)) ** 5
        lower = max(.001, scaled.mean() - 2 * scaled.std())
        upper = min(10., scaled.mean() + 2 * scaled.std())
        expected = np.clip(.1 + .9 * scaled, lower, upper)
        policy.algorithm.update_alpha_balance()
        self.assertAlmostEqual(policy.algorithm.alpha_balance['text'], expected[0])
        self.assertAlmostEqual(policy.algorithm.alpha_balance['agentic'], expected[1])
        self.assertEqual(policy.algorithm.alpha_balance['single'], 1.)
        self.assertEqual(len(scope['_OUTPUT'].tracked_hit_stats['text']), 1)

    def test_arc_b1_then_b2_feedback_changes_target_and_frequency_queue(self):
        policy, _ = make('ARC', capacity=2)
        create(policy, 0)
        create(policy, 1)
        access(policy, 0, 1.)
        access(policy, 1, 2.)
        access(policy, 1, 3.)
        eligible(policy, 0)
        eligible(policy, 1)
        self.assertEqual(policy.algorithm.free_table[1].access_count, 2)
        self.assertEqual(len(policy.algorithm.T1), 1)
        self.assertEqual(len(policy.algorithm.T2), 1)
        self.assertEqual(policy.evict({'now': 4.})['block_id'], 0)
        self.assertIn('0', policy.algorithm.B1)
        create(policy, 0, generation=2)
        event(policy, 'lookup_completed', time=5., content_hash='0', hit=False)
        self.assertEqual(policy.algorithm.p, 1)
        self.assertFalse(policy.algorithm.miss_in_B2)
        access(policy, 0, 6.)
        eligible(policy, 0)
        self.assertEqual(policy.evict({'now': 7.})['block_id'], 1)
        self.assertIn('1', policy.algorithm.B2)
        event(policy, 'lookup_completed', time=8., content_hash='1', hit=False)
        self.assertEqual(policy.algorithm.p, 0)
        self.assertTrue(policy.algorithm.miss_in_B2)
        self.assertNotIn('1', policy.algorithm.B2)

    def test_lecar_rng_matches_seeded_numpy_reference(self):
        policy, _ = make('LeCaR')
        expected = [0 if value < .5 else 1 for value in np.random.RandomState(123).rand(20)]
        self.assertEqual([policy.algorithm.getChoice() for _ in range(20)], expected)

    def test_lecar_both_expert_ghost_penalties_match_discounted_float32_math(self):
        policy, _ = make('LeCaR', capacity=2)
        create(policy, 0)
        create(policy, 1)
        access(policy, 0, .5)
        access(policy, 0, 1.)
        access(policy, 1, 2.)
        eligible(policy, 0)
        eligible(policy, 1)
        # LRU nominates 0, LFU nominates 1. Seed 123 first chooses LFU.
        self.assertEqual(policy.evict({'now': 3.})['block_id'], 1)
        self.assertIn('1', policy.algorithm.LFU_HIST)
        expected = expected_penalty(policy.algorithm.W.copy(), 1, policy.algorithm.discount_rate, 2.)
        event(policy, 'lookup_completed', time=5., content_hash='1', hit=False)
        np.testing.assert_allclose(policy.algorithm.W, expected, rtol=0, atol=6e-8)
        self.assertLess(policy.algorithm.W[1], .5)
        self.assertNotIn('1', policy.algorithm.LFU_HIST)
        create(policy, 1, generation=2)
        access(policy, 1, 6.)
        eligible(policy, 1)
        # Second random draw chooses LRU, which still nominates old/frequent 0.
        self.assertEqual(policy.evict({'now': 8.})['block_id'], 0)
        self.assertIn('0', policy.algorithm.LRU_HIST)
        expected = expected_penalty(policy.algorithm.W.copy(), 0, policy.algorithm.discount_rate, 1.)
        event(policy, 'lookup_completed', time=9., content_hash='0', hit=False)
        np.testing.assert_allclose(policy.algorithm.W, expected, rtol=0, atol=6e-8)
        self.assertNotIn('0', policy.algorithm.LRU_HIST)
        self.assertAlmostEqual(float(policy.algorithm.W.sum()), 1., places=6)

    def test_lecar_expert_weight_floor_and_ceiling(self):
        policy, _ = make('LeCaR')
        for _ in range(100):
            policy.algorithm.adjustWeights(-1., 0.)
        np.testing.assert_array_equal(policy.algorithm.W, np.array([.01, .99], dtype=np.float32))
        for _ in range(100):
            policy.algorithm.adjustWeights(0., -1.)
        np.testing.assert_array_equal(policy.algorithm.W, np.array([.99, .01], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
