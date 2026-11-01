"""Private native-versus-v2 shadow audit; never shipped to a coding workspace.

Native algorithms continue to control the simulator. The isolated candidate receives
only public v2 observations and must independently choose the identical victim.
"""
import inspect
import json
import os
from pathlib import Path

from policy_agent.bridge import PolicyFailure
from policy_agent.v2_bridge import IndependentEvictor


class NativeShadowEvictor(IndependentEvictor):
    def __init__(self, base, source):
        self.base = base
        super().__init__(source)
        self.checked = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __contains__(self, block_id):
        return block_id in self.base

    @property
    def num_blocks(self):
        if self.base.num_blocks != len(self.eligible):
            raise PolicyFailure('Shadow eligibility bookkeeping differs from native membership')
        return self.base.num_blocks

    def add(self, *args, **kwargs):
        data = dict(inspect.signature(self.base.add).bind(*args, **kwargs).arguments)
        self.base.add(*args, **kwargs)
        # Only physical facts reach the independent implementation. Native counts,
        # timestamps, classifications and scores are deliberately discarded here.
        super().add(data['block_id'], data['content_hash'], data['num_hashed_tokens'], None)

    def remove(self, block_id):
        self.base.remove(block_id)
        super().remove(block_id)

    def update(self, *args, **kwargs):
        self.base.update(*args, **kwargs)

    def evict(self):
        self.decision_id += 1
        preceding = self.events[-32:]
        candidate = self.flush(True)
        block_id, generation = candidate['block_id'], candidate['generation']
        if (block_id not in self.eligible or block_id not in self.base
                or generation != self.generations.get(block_id)):
            raise PolicyFailure(f'Shadow candidate returned stale/ineligible victim: {candidate}')
        native_id, native_hash = self.base.evict()
        candidate_hash = self.eligible[block_id]
        if native_id != block_id or native_hash != candidate_hash:
            # Kept entirely on the trusted side. This diagnostic is never fed to
            # the policy process or exposed as agent submission feedback.
            diagnostic = {
                'cache_id': self.cache_id, 'decision_id': self.decision_id,
                'native': {'block_id': native_id, 'content_hash': str(native_hash)},
                'candidate': candidate, 'preceding_events': preceding,
                'native_state': self._native_diagnostic(native_id, block_id),
            }
            path = os.environ.get('UNICACHE_POLICY_SHADOW_FAILURE')
            if path:
                Path(path).write_text(json.dumps(diagnostic, indent=2, default=str))
            raise PolicyFailure('ShadowPolicyMismatch at decision '
                                f'{self.decision_id}: native={native_id}, candidate={block_id}; '
                                + json.dumps(diagnostic, default=str)[-12000:])
        self.checked += 1
        del self.eligible[native_id]
        self.emit({'kind': 'block_evicted', 'block_id': native_id,
                   'generation': generation, 'content_hash': str(native_hash),
                   'decision_id': self.decision_id})
        path = os.environ.get('UNICACHE_POLICY_DECISIONS')
        if path:
            if self.audit is None:
                self.audit = open(path, 'a')
            self.audit.write(json.dumps({'cache_id': self.cache_id,
                                         'decision_id': self.decision_id,
                                         'block_id': native_id,
                                         'content_hash': str(native_hash)}) + '\n')
        return native_id, native_hash

    def _native_diagnostic(self, *ids):
        result = {'type': type(self.base).__name__}
        for name in ('p', 'miss_in_B2', 'alpha_balance', 'W'):
            if hasattr(self.base, name):
                value = getattr(self.base, name)
                result[name] = value.tolist() if hasattr(value, 'tolist') else value
        for name in ('T1', 'T2', 'B1', 'B2', 'LRU', 'LFU', 'priority_queue'):
            value = getattr(self.base, name, None)
            if value is not None:
                if isinstance(value, dict):
                    result[name] = {str(k): len(v) if hasattr(v, '__len__') else str(v)
                                    for k, v in list(value.items())[:16]}
                else:
                    result[name] = {'length': len(value), 'head': str(value[:3])}
        table = getattr(self.base, 'free_table', {})
        result['blocks'] = {str(i): vars(table[i]) for i in ids if i in table}
        return result
