"""Small public v2 lifecycle exercise. Uses no workload trace or baseline scores."""
import json
from pathlib import Path
from policy import Policy

context = json.loads(Path('run_context.json').read_text())
config = {'api_version': 2, 'capacity': context['gpu_blocks'], 'block_size': 8,
          'cache_id': 'gpu:0', 'clock_unit': 'simulated_seconds'}
p = Policy(config)
eligible = {1:1, 2:1}
ordinal = 0


def emit(kind, **fields):
    global ordinal
    ordinal += 1
    p.on_event({'kind':kind, 'event_id':ordinal, 'time':1., 'cache_id':'gpu:0', **fields})


for i in eligible:
    emit('block_allocated', block_id=i, generation=1)
    emit('hash_binding', block_id=i, content_hash=str(100+i), depth=i*8)
    step = {'sequence_id':i,'request_id':str(i),'block_ids':[i], 'time':1., 'now':1.,
            'is_prefill':True,'workload_class':'ShareGPT-multi-turn_test',
            'prompt_tokens_len':8, 'output_tokens_len':0}
    emit('sequence_step', **{k:v for k,v in step.items() if k != 'time'})
    emit('sequence_release', sequence_id=i, block_ids=[i], last_step=step)
    emit('eligibility_changed', block_id=i, generation=1, content_hash=str(100+i),
         depth=i*8, eligible=True)

for decision_id in (1,2):
    victim = p.evict({'cache_id':'gpu:0','now':2., 'decision_id':decision_id,
                      'through_event_id':ordinal,'eligible_count':len(eligible)})
    if not isinstance(victim,dict) or set(victim) != {'block_id','generation'}:
        raise TypeError('Return exactly block_id and generation; no None fallback')
    b = victim['block_id']
    if type(b) is not int or b not in eligible or victim['generation'] != eligible[b]:
        raise ValueError('Invalid, ineligible or stale victim')
    del eligible[b]
    emit('block_evicted', **victim, content_hash=str(100+b), decision_id=decision_id)
print('PASS: v2 lifecycle, generations and repeated legal decisions')
print('Smoke test passed. No traces used; no submission consumed.')
