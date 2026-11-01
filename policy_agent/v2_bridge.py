"""Independent v2 evictor: trusted eligibility, isolated editable policy state."""
import atexit
import enum
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from policy_agent.bridge import PolicyFailure, Remote, profile


def numerical_runtime():
    """Expose vetted numerical packages, not the entire environment or repository."""
    roots = list(dict.fromkeys([sysconfig.get_path('purelib'), sysconfig.get_path('platlib')]))
    allowed = []
    versions = {}
    for name in ('numpy', 'scipy', 'charset_normalizer'):
        versions[name] = importlib.metadata.version(name)
        for root in roots:
            allowed.extend(Path(root).glob(name + '*'))
    return roots, [str(p.resolve()) for p in allowed], versions


class V2Remote(Remote):
    def __init__(self, source, config, *, mode_only=False):
        worker = Path(__file__).with_name('v2_worker.py').resolve()
        paths, packages, versions = numerical_runtime()
        runtime = Path(sys.executable).resolve()
        rules = profile([worker, sys.base_prefix, runtime, *packages])
        self.process = subprocess.Popen(
            ['/usr/bin/sandbox-exec', '-p', rules, str(runtime), '-I', '-S', '-c', worker.read_text()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            env={'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'VECLIB_MAXIMUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
        try:
            self.initial = self.call({'source': source, 'config': {**config, 'dependencies': versions},
                                      'dependency_paths': paths, 'mode_only': mode_only})
        except BaseException:
            self.close()
            raise


def serializable(value):
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


class NativeDecisionAudit:
    """Private parity instrumentation; forwards all native behavior unchanged."""
    ordinal = 0

    def __init__(self, base):
        self.base = base
        self.cache_id = 'gpu:0' if NativeDecisionAudit.ordinal == 0 else 'cpu:0'
        NativeDecisionAudit.ordinal += 1
        self.decision_id = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __contains__(self, block_id):
        return block_id in self.base

    def evict(self):
        victim = self.base.evict()
        self.decision_id += 1
        with open(os.environ['UNICACHE_POLICY_DECISIONS'], 'a') as out:
            out.write(json.dumps({'cache_id': self.cache_id, 'decision_id': self.decision_id,
                                  'block_id': victim[0], 'content_hash': str(victim[1])}) + '\n')
        return victim


class IndependentEvictor:
    """No native base evictor, no strategy switch and no None delegation."""
    def __init__(self, source):
        self.source = source
        self.remote = None
        self.eligible = {}
        self.generations = {}
        self.events = []
        self.cache_id = 'gpu:0'
        self.config = {}
        self.event_id = 0
        self.decision_id = 0
        self.audit = None
        atexit.register(self.close)
        from policy_agent.events import register
        register(self)

    def close(self):
        if self.remote is not None:
            self.remote.close()
            self.remote = None
        if self.audit is not None:
            self.audit.close()
            self.audit = None

    def __contains__(self, block_id):
        return block_id in self.eligible

    @property
    def num_blocks(self):
        return len(self.eligible)

    def add(self, block_id, content_hash, num_hashed_tokens, last_accessed):
        if block_id not in self.generations:
            raise PolicyFailure('Missing physical generation observation before eligibility')
        self.eligible[block_id] = content_hash
        self.emit({'kind': 'eligibility_changed', 'block_id': block_id,
                   'generation': self.generations[block_id],
                   'content_hash': str(content_hash), 'depth': num_hashed_tokens,
                   'eligible': True, 'reference_count': 0, 'reason': 'release'})

    def remove(self, block_id):
        if block_id not in self.eligible:
            raise PolicyFailure('Host tried to remove an ineligible block')
        del self.eligible[block_id]
        self.emit({'kind': 'eligibility_changed', 'block_id': block_id,
                   'generation': self.generations.get(block_id, 0),
                   'eligible': False, 'reason': 'reuse'})

    def update(self, *args, **kwargs):
        # LRU scaffold metadata is deliberately not used by policy selection.
        pass

    def emit(self, event):
        event = serializable(event)
        if 'time' not in event:
            from vllm.simulator.timer_utils import GLOBAL_TIMER
            event['time'] = GLOBAL_TIMER.get_current_time()
        self.event_id = max(self.event_id + 1, event.get('event_id', 0))
        event['event_id'] = self.event_id
        event.setdefault('cache_id', self.cache_id)
        block_id = event.get('block_id')
        if block_id is not None and 'generation' in event:
            self.generations[block_id] = event['generation']
        self.events.append(event)
        if len(self.events) >= 2048:
            self.flush(False)

    def _start(self):
        if self.remote is not None:
            return
        from vllm.simulator.config import GLOBAL_SIMULATOR_CONFIG
        cfg = GLOBAL_SIMULATOR_CONFIG
        config = {'api_version': 2, 'event_semantics_version': 2, 'cache_id': self.cache_id,
                  'gpu_blocks': cfg.simulator_num_gpu_blocks,
                  'cpu_blocks': cfg.simulator_num_cpu_blocks, 'block_size': 8,
                  'wa_fit_interval': cfg.simulator_wa_fit_interval,
                  'seed': 0, 'clock_unit': 'simulated_seconds', **self.config}
        self.remote = V2Remote(self.source, config)
        if self.remote.initial['cache_mode'] != 'prefix':
            raise PolicyFailure('Disabled-cache policy must be configured before engine construction')

    def initialize(self):
        self._start()

    def flush(self, choose):
        self._start()
        from vllm.simulator.timer_utils import GLOBAL_TIMER
        context = {'cache_id': self.cache_id, 'now': GLOBAL_TIMER.get_current_time(),
                   'decision_id': self.decision_id, 'through_event_id': self.event_id,
                   'eligible_count': len(self.eligible), 'reason': 'allocation'}
        result = self.remote.call({'events': self.events, 'choose': choose, 'context': context})
        if result['through_event_id'] != self.event_id:
            raise PolicyFailure('Worker did not acknowledge the complete event batch')
        self.events.clear()
        return result['victim']

    def evict(self):
        self.decision_id += 1
        victim = self.flush(True)
        if (not isinstance(victim, dict) or set(victim) != {'block_id', 'generation'}
                or any(type(value) is not int for value in victim.values())):
            raise PolicyFailure('InvalidEvictionError: victim must contain integer block_id and generation')
        block_id, generation = victim['block_id'], victim['generation']
        if block_id not in self.eligible or generation != self.generations.get(block_id):
            raise PolicyFailure(f'InvalidEvictionError: ineligible/stale victim {victim}')
        content_hash = self.eligible.pop(block_id)
        self.emit({'kind': 'block_evicted', 'block_id': block_id, 'generation': generation,
                   'content_hash': str(content_hash), 'decision_id': self.decision_id})
        path = os.environ.get('UNICACHE_POLICY_DECISIONS')
        if path:
            if self.audit is None:
                self.audit = open(path, 'a')
            self.audit.write(json.dumps({'cache_id': self.cache_id, 'decision_id': self.decision_id,
                                         'block_id': block_id, 'content_hash': str(content_hash)}) + '\n')
        return block_id, content_hash
