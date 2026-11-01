"""Editable raw-event bookkeeping copied into standalone v2 reference sources.

This file deliberately has no simulator imports. The source generator prepends only
selected algorithm definitions and public numerical dependencies. No hidden native
baseline is consulted at runtime.
"""


class _Clock:
    now = 0.0
    def get_current_time(self):
        return self.now


class _KnownRequests:
    def __init__(self):
        self.reuse_track = {}
        self.counter = 0
        self.prefill_counter = 0

    def admitted(self, hashes):
        for h in hashes:
            self.reuse_track.setdefault(str(h), []).append(self.counter)
        self.counter += 1

    def _get_next_use(self, content_hash):
        times = self.reuse_track.get(str(content_hash), [])
        while times and times[0] <= self.prefill_counter:
            times.pop(0)
        return times[0] if times else float('inf')


def get_global_output_simulator():
    return _OUTPUT


def _category(key, random_class=False):
    if key.startswith(('ShareGPT-multi-turn', 'text_ShareGPT-multi-turn')):
        return TaskType.MULTI_TURN_TEXT
    for prefix, task in [('image_', TaskType.MULTI_TURN_IMAGE),
                         ('search_', TaskType.MULTI_TURN_SEARCH),
                         ('file_', TaskType.MULTI_TURN_FILE)]:
        if key.startswith(prefix + 'ShareGPT-multi-turn'):
            return task
    if key.startswith('alfred_'):
        return TaskType.MULTI_TURN_AGENTIC
    if random_class and key.startswith(('apps_', 'mind2web_')):
        return TaskType.SINGLE_TURN_RANDOM
    return TaskType.SINGLE_TURN


class Policy:
    CACHE_MODE = 'disabled' if BASELINE == 'NONE' else 'prefix'

    def __init__(self, config):
        global GLOBAL_TIMER, GLOBAL_SIMULATOR_CONFIG, GLOBAL_WA_HELPER
        global GLOBAL_ONLINE_OPT_HELPER, _OUTPUT
        self.config = config
        GLOBAL_TIMER = _Clock()
        GLOBAL_SIMULATOR_CONFIG = SimpleNamespace(
            eviction_policy='SimpleHYBRID-OPT' if BASELINE == 'SimpleHYBRID-OPT' else BASELINE,
            simulator_enable_virtual_time=True,
            simulator_num_gpu_blocks=config.get('gpu_blocks', config.get('capacity', config.get('num_blocks', 50000))),
            tail_first=BASELINE == 'OFFSET-TAIL', head_first=BASELINE == 'OFFSET-HEAD')
        if BASELINE in ('OPT-Online', 'SimpleHYBRID-OPT'):
            GLOBAL_ONLINE_OPT_HELPER = _KnownRequests()
        else:
            GLOBAL_ONLINE_OPT_HELPER = None
        if BASELINE in ('UniCache', 'SimpleHYBRID-OPT'):
            _OUTPUT = SimpleNamespace(step_track=0, tracked_hit_stats={
                q: deque(maxlen=500) for q in ['single','text','file','image','search','agentic']})
        else:
            _OUTPUT = None
        if BASELINE == 'WA':
            GLOBAL_WA_HELPER = WAHelper(config.get('wa_fit_interval', 1200))
        classes = {'LRU': 'LRUEvictor', 'NONE': 'LRUEvictor', 'LFU': 'LFUEvictor',
                   'FIFO': 'FIFOEvictor', 'ARC': 'ARCEvictor', 'LeCaR': 'LeCaREvictor',
                   'WA': 'WAEvictor', 'UniCache': 'SimpleHYBRIDEvictor',
                   'SimpleHYBRID-OPT': 'SimpleHYBRIDEvictor',
                   'OPT-Online': 'OPTEvictor', 'OFFSET-HEAD': 'OffsetEvictor',
                   'OFFSET-TAIL': 'OffsetEvictor'}
        self.algorithm = globals()[classes[BASELINE]]()
        self.blocks = {}
        self.steps = {}
        self.block_size = config.get('block_size', 8)
        self.known_admissions = set()

    def _block(self, block_id):
        return self.blocks.setdefault(block_id, {
            'generation': 0, 'content_hash': None, 'depth': 0, 'active': True,
            'last': -1, 'first': -1, 'count': 0, 'last_hit': -1,
            'task': TaskType.DEFAULT, 'is_decode': None})

    def _reset_active(self, b, new_content=False):
        b['active'] = True
        b['last'] = -1
        if new_content or BASELINE in ('ARC', 'LeCaR', 'UniCache', 'SimpleHYBRID-OPT'):
            b.update(count=0, task=TaskType.DEFAULT, is_decode=None)
        if new_content:
            b.update(first=-1, last_hit=-1)

    def _access(self, ids, now, step):
        task = _category(step.get('workload_class', ''), BASELINE in ('UniCache', 'SimpleHYBRID-OPT'))
        out = step.get('output_tokens_len', 0)
        decode_n = (out - 1) // self.block_size + 1 if out else 0
        for i, block_id in enumerate(ids):
            b = self._block(block_id)
            eligible = block_id in self.algorithm
            is_decode = decode_n > 0 and i >= len(ids) - decode_n
            if BASELINE in ('LRU', 'OFFSET-HEAD', 'OFFSET-TAIL', 'OPT-Online', 'NONE'):
                if eligible:
                    self.algorithm.update(block_id, now)
                else:
                    b['last'] = now
            elif BASELINE in ('ARC', 'LeCaR'):
                if eligible:
                    self.algorithm.update(block_id, now)
                else:
                    b['last'] = now
                    b['count'] += 1
            elif BASELINE in ('UniCache', 'SimpleHYBRID-OPT'):
                if eligible:
                    self.algorithm.update(block_id, now, task, is_decode)
                else:
                    b.update(last=now, task=task, is_decode=is_decode)
                    if not is_decode:
                        b['count'] += 1
            elif BASELINE == 'FIFO':
                if b['first'] == -1:
                    b['first'] = now
            elif BASELINE == 'LFU':
                b['count'] += 1
                if eligible:
                    self.algorithm.update(block_id, b['count'])
            elif BASELINE == 'WA':
                b.update(last=now, task=task)
                if step.get('is_prefill', False):
                    if b['last_hit'] != -1:
                        GLOBAL_WA_HELPER.collect_hit_interval(task, now-b['last_hit'])
                    b['last_hit'] = now

    def _eligible(self, block_id, event):
        b = self._block(block_id)
        b['content_hash'] = str(event.get('content_hash', b['content_hash']))
        b['depth'] = event.get('depth', event.get('num_hashed_tokens', b['depth']))
        b['generation'] = event.get('generation', b['generation'])
        args = [block_id, b['content_hash'], b['depth']]
        if BASELINE == 'LFU':
            args += [b['count']]
        elif BASELINE == 'FIFO':
            args += [b['first']]
        elif BASELINE in ('ARC', 'LeCaR'):
            args += [b['last'], b['count']]
        elif BASELINE in ('UniCache', 'SimpleHYBRID-OPT'):
            args += [b['last'], b['count'], b['task'], b['is_decode']]
        elif BASELINE == 'WA':
            args += [b['last'], b['task']]
        else:
            args += [b['last']]
        self.algorithm.add(*args)
        b['active'] = False
        b['last'] = -1
        if BASELINE in ('ARC', 'LeCaR', 'UniCache', 'SimpleHYBRID-OPT'):
            b.update(count=0, task=TaskType.DEFAULT, is_decode=None)

    def on_event(self, event):
        GLOBAL_TIMER.now = event.get('time', event.get('now', GLOBAL_TIMER.now))
        kind = event['kind']
        block_id = event.get('block_id')
        if kind == 'cache_reset':
            # Match native evictor replacement. Global fits, hit windows, queued
            # demand and admission/progress counters are not physical-cache state.
            self.algorithm = type(self.algorithm)()
            self.blocks.clear()
            self.steps.clear()
        elif kind in ('block_allocated', 'block_created'):
            b = self._block(block_id)
            self._reset_active(b, True)
            b['generation'] = event.get('generation', b['generation'] + 1)
        elif kind in ('hash_binding', 'block_changed'):
            b = self._block(block_id)
            if event.get('content_hash') is not None:
                b['content_hash'] = str(event['content_hash'])
            b['depth'] = event.get('depth', event.get('prefix_tokens', b['depth']))
        elif kind in ('activation', 'active', 'block_activated') and event.get('refcount', 1) == 1:
            if block_id in self.algorithm:
                self.algorithm.remove(block_id)
            self._reset_active(self._block(block_id))
        elif kind in ('eligible', 'eligibility_changed'):
            if event.get('eligible', True):
                self._eligible(block_id, event)
            else:
                if block_id in self.algorithm:
                    self.algorithm.remove(block_id)
                self._reset_active(self._block(block_id))
        elif kind == 'sequence_step':
            self.steps[event.get('sequence_id', event.get('seq_id'))] = event
            if BASELINE == 'WA' or BASELINE == 'FIFO' or (BASELINE == 'LFU' and event['is_prefill']):
                self._access(event['block_ids'], event.get('now', event.get('time', GLOBAL_TIMER.now)), event)
        elif kind == 'sequence_release':
            step = event.get('last_step') or self.steps.get(event.get('sequence_id', event.get('seq_id')))
            if step and BASELINE not in ('WA', 'FIFO', 'LFU'):
                self._access(event['block_ids'], step.get('now', step.get('time', GLOBAL_TIMER.now)), step)
            self.steps.pop(event.get('sequence_id', event.get('seq_id')), None)
        elif kind == 'lookup_completed' and event.get('miss', not event.get('hit', True)):
            if BASELINE in ('ARC', 'LeCaR'):
                self.algorithm.miss(str(event['content_hash']))
        elif kind == 'prefill_accounted' and BASELINE in ('UniCache','SimpleHYBRID-OPT'):
            _OUTPUT.step_track = event.get('accounting_ordinal', event.get('ordinal', _OUTPUT.step_track+1))
            category = event.get('category', event.get('task_category', 'single'))
            _OUTPUT.tracked_hit_stats[category].append(event.get('hit_tokens', event.get('cached_prompt_tokens', 0)))
            for q, values in _OUTPUT.tracked_hit_stats.items():
                if q != category:
                    old = list(values)
                    values.clear()
                    values.extend(x * .95 for x in old)
        elif kind == 'request_admitted' and BASELINE in ('OPT-Online','SimpleHYBRID-OPT'):
            rid = event.get('request_id', event.get('ordinal'))
            if rid not in self.known_admissions:
                GLOBAL_ONLINE_OPT_HELPER.admitted(event.get('prefix_hashes', []))
                self.known_admissions.add(rid)
        elif kind in ('engine_iteration', 'prefill_progress') and BASELINE in ('OPT-Online','SimpleHYBRID-OPT'):
            GLOBAL_ONLINE_OPT_HELPER.prefill_counter = event.get('prefill_counter', event.get('prefill_counter_after', 0))
        elif kind == 'block_evicted':
            # Selection already removed algorithm membership, just as native evict.
            pass

    def evict(self, context):
        GLOBAL_TIMER.now = context['now']
        block_id, _ = self.algorithm.evict()
        return {'block_id': block_id, 'generation': self.blocks[block_id]['generation']}
