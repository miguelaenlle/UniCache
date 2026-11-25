# SPDX-License-Identifier: Apache-2.0
"""Process-wide KV-cache contention counters (sim2real validity checks).

evictions: cached blocks evicted to make room.
evicted_then_missed: lookup misses on content that had been evicted earlier,
    i.e. evictions that cost a later hit. Contention is real when this is > 0.
preemptions: requests preempted by the scheduler (recompute).
Counters only observe; they never change scheduling or eviction.
"""
CACHE_STATS = {"evictions": 0, "evicted_then_missed": 0, "preemptions": 0}
_EVICTED_HASHES = set()


def record_eviction(content_hash):
    CACHE_STATS["evictions"] += 1
    _EVICTED_HASHES.add(content_hash)


def record_miss(content_hash):
    if content_hash in _EVICTED_HASHES:
        CACHE_STATS["evicted_then_missed"] += 1


def record_preemption():
    CACHE_STATS["preemptions"] += 1
