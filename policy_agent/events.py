"""Ordered online facts for the independent v2 policy worker.

This module transports physical/request observations only; it owns no retention rule.
Native runs incur no observation calls unless POLICY_AGENT_INTERFACE=v2 is selected.
"""
import os

_observers = []
_ordinal = 0


def independent_enabled():
    return (os.environ.get('UNICACHE_POLICY_INTERFACE') == 'v2'
            or os.environ.get('POLICY_AGENT_INTERFACE') == 'v2')


def enabled():
    return independent_enabled() or os.environ.get('UNICACHE_POLICY_OBSERVE') == '1'


def register(observer):
    # CpuGpuBlockAllocator creates GPU then CPU. It binds explicit IDs before use.
    if not getattr(observer, 'cache_id', None):
        observer.cache_id = 'gpu:0' if not _observers else 'cpu:0'
    _observers.append(observer)


def configure(**fields):
    """Bind public serving context before any worker receives observations."""
    if enabled():
        for observer in tuple(_observers):
            observer.config.update(fields)


def unregister(observer):
    if observer in _observers:
        _observers.remove(observer)


def emit(kind, cache_id=None, **fields):
    if not enabled():
        return
    global _ordinal
    from vllm.simulator.timer_utils import GLOBAL_TIMER
    _ordinal += 1
    event = {'kind': kind, 'event_id': _ordinal,
             'time': GLOBAL_TIMER.get_current_time(), **fields}
    if cache_id is not None:
        event['cache_id'] = cache_id
    for observer in tuple(_observers):
        if cache_id is None or getattr(observer, 'cache_id', None) == cache_id:
            observer.emit(dict(event))


def emit_to(observer, kind, **fields):
    if enabled():
        emit(kind, cache_id=observer.cache_id, **fields)


def workload_class(request_id):
    """Return only the current request's online workload identifier."""
    from vllm.simulator.global_output_simulator import get_global_output_simulator
    output = get_global_output_simulator()
    conv = output.request_id_2_conv_key_map.get(request_id)
    return output.parser_conv_key(conv)[0] if conv is not None else ''
