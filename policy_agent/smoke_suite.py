"""Trusted synthetic checks, executed in the same isolation as custom local tests."""
import inspect
from policy import Policy


def check(events, eligible, now):
    policy = Policy()
    if not callable(getattr(policy,'on_event',None)) or not callable(getattr(policy,'evict',None)):
        raise TypeError('Policy must implement on_event(event) and evict(now)')
    inspect.signature(policy.on_event).bind({})
    inspect.signature(policy.evict).bind(now)
    for event in events:
        policy.on_event(event)
    while eligible:
        decision = policy.evict(now)
        if decision is None:
            # Delegation is valid. Simulate removal, not the real starting policy's ranking.
            decision = min(eligible)
        if type(decision) is not int or decision not in eligible:
            raise ValueError(f'InvalidEvictionError: Block {decision!r} is not eligible for eviction')
        eligible.remove(decision)
        policy.on_event({'kind':'remove','block_id':decision})


def block(i):
    return {'kind':'add','block_id':i,'content_hash':100+i,'num_hashed_tokens':8*i,
            'last_accessed':float(i),'access_count':i,'task_type':'MULTI_TURN_TEXT','is_decode':False}

check([block(1)], {1}, 2.)
print('PASS: interface and single eligible block')
check([block(1),block(2), {'kind':'update','block_id':1,'last_accessed':4.},
       {'kind':'remove','block_id':2}], {1}, 5.)
print('PASS: updates and removal of ineligible blocks')
check([block(i) for i in range(1,6)], set(range(1,6)), 6.)
print('PASS: repeated legal eviction decisions')
print('Smoke test passed. No trace data used; no submission attempt consumed.')
