# Policy interface v2: implement the online baselines from scratch

Status: v2 is implemented in `f1b4309`, with the source-generation correction in
`cd2e82e`; reference validation
is recorded in [the verification report](result/policy-interface-v2/report.md).
The sections below retain the design rationale. The implemented field names and
commands are documented in [v2_api.md](policy_agent/v2_api.md) and
[EXPERIMENTS.md](EXPERIMENTS.md). Existing v1 agent experiments remain separate.

Implementation detail: the physical allocator uses one fixed LRU scaffold for every
v2 policy. Its recency metadata is ignored by the independent worker, which has no
native backing evictor. The editable source owns all algorithm bookkeeping and every
victim decision. Private native-shadow runs are a separate validation mode.

## Goal and scope

Give every agent policy the same observations and control, independent of its starting
policy. The agent must own its policy state and choose every eviction. The allocator
continues enforcing cache correctness.

The interface must express LRU, LFU, FIFO, ARC, LeCaR, WA, UniCache (`SimpleHYBRID`),
OFFSET-HEAD, OFFSET-TAIL, OPT-Online and UniCache-OPT (`SimpleHYBRID-OPT`). Offline OPT
remains a private, privileged comparison and is excluded from the agent interface.
NONE is a separate cache-disabled control, addressed explicitly below.

“Can express” and “will rediscover” are different claims. Private reference ports and
parity tests establish expressibility. Agent optimization establishes whether the
coding model can discover effective rules without receiving those reference ports.

## Existing mechanism and its limitations

| Component | Current behavior | Planned modification |
|---|---|---|
| `vllm/core/evictor.py` | `make_evictor` builds the selected baseline. Our wrapper optionally intercepts its calls. Baselines maintain queues, histories, scores and weights. | Add a distinct agent-owned evictor implementation, not a wrapper around a baseline. Leave native baselines available as references. |
| `vllm/core/block/prefix_caching_block.py` | Selects policy-specific tracker classes. Updates counts/timestamps/categories differently. Calls `add` when cached blocks reach reference count zero, and `remove` when reused. | Emit uniform allocation, hash, lookup and reference-lifecycle facts. Keep physical allocation, ownership, reference counts, computed flags and eligibility trusted. Move policy-specific counters and reset choices into policy code. |
| `vllm/core/block_manager.py` | LRU/ARC/LeCaR/UniCache defer sequence-access updates until sequence release; FIFO and LFU use different update paths. | Emit sequence-step and release observations regardless of policy. Include final block tables and the last step's metadata so reference policies can reproduce deferred updates. |
| `vllm/core/block/cpu_gpu_block_allocator.py` | Forwards policy-specific calls to device allocators. | Forward uniform events with cache/device identity. |
| `vllm/core/scheduler.py` | Supplies sequence-step information and records prefix-hit accounting. | Publish the same step and first-prefill accounting events for every policy. |
| `vllm/simulator/output_simulator.py` | `step_track` counts first recorded requests. Keeps per-task hit windows, decaying inactive categories. UniCache reads this global object directly. | Publish raw ordered accounting observations; v2 policies build their own windows, decay and weights. Trusted output scoring remains separate. |
| `vllm/engine/llm_engine.py`, `vllm/simulator/opt_utils.py` | Online variants observe requests after scheduler admission and maintain a separate prefill-iteration counter. Offline OPT reads a full future-use map. | Publish admitted-request prefixes and engine/prefill boundaries to everyone. Never publish the offline map. |
| `policy_agent/bridge.py` | Serializes arguments of the starting evictor's `add/update/remove`. `None` calls the hidden baseline. Other methods, including `miss`, fall through to that baseline. | Replace argument introspection and fall-through with a versioned event protocol and mandatory agent decisions. |
| `policy_agent/worker.py`, `starter.py`, `smoke_suite.py` | `Policy()` receives no config; default `evict` returns `None`. | Use `Policy(config)`, validate returned block generation, expand synthetic tests, and seed actual editable policy implementations. |

Two implementation details must not be glossed over:

1. **Miss ordering:** today, `allocate_immutable_block` detects a miss, allocates/appends
   the replacement block, and only then calls ARC/LeCaR `miss`. Allocation may already
   have evicted something. Earlier discussion incorrectly assumed this notification
   preceded eviction. Expose the actual before/after boundaries. Reference ports must
   preserve native ordering; changing the native algorithm's order is a separate study.
2. **Deferred bookkeeping:** a scheduler step is not the same as an immediate update
   to every baseline's block counters. Emit facts sufficient to reconstruct both
   immediate and deferred behavior, rather than inventing one universal access count.

## Policy code and decisions

```python
class Policy:
    CACHE_MODE = "prefix"  # Optional "disabled" supports the NONE control.

    def __init__(self, config):
        self.config = config
        # All queues, counters, distributions, histories and weights belong here.

    def on_event(self, event):
        pass

    def evict(self, context):
        # Must choose an eligible block in context["cache_id"].
        return {"block_id": 42, "generation": 3}
```

No `None` delegation and no imports of native baseline implementations. A starting
policy supplies real editable code implementing this contract. For example, an LRU
start contains its own recency tracking and victim selection. Other baseline source
is not automatically placed in the coding workspace.

For a reference port, constants such as UniCache's 600-second decay scale and
2,048-token offset normalization, or LeCaR's learning rate, belong in that policy's
code. Do not apply them invisibly in the host or put baseline answers into config.
An agent may choose or learn different parameters.

### Initialization: read-only environment context

Expose a JSON-serializable configuration with:

- `api_version` and documented `event_semantics_version`.
- Cache IDs, devices, capacity in blocks, block size in tokens, KV dtype and bytes per block.
- Model identifier and relevant architecture dimensions; no model weights required.
- Virtual clock units and origin; time is simulated seconds, never wall-clock test duration.
- Fixed scheduler settings relevant to requests: maximum sequences, token/context limits,
  preemption mode and the active timing-model identifier/parameters.
- Available task-category values and definitions, including `single_turn_random`.
- Experiment/runtime random seed and resource/dependency limits.
- An explicit declaration that unseen arrivals and the offline OPT map are unavailable.

Do not include selected baseline names/scores (except the requested starting policy),
eval identifiers, future session/turn counts, future gaps, or reference-policy parameters.
Do not expose `max_tokens` when it is merely the trace's known future response length;
only expose a request's declared output limit if that information is genuinely part
of the online serving contract. Current generated output length is observable.

### Identity and common event envelope

Every event carries `event_id`, `time`, `kind`, and applicable `cache_id`, `request_id`
and `sequence_id`. Event IDs are monotonically increasing within one evaluation;
ordering is defined even when timestamps tie.

A physical block is identified by `(cache_id, block_id, generation)`. Increment its
generation when the physical slot is assigned new content. A `content_hash` is an
opaque, stable prefix identity serialized losslessly as a string; it is not an
arithmetic feature. Prefix depth (`prefix_tokens`) differs from block size.

Mutable blocks can initially have `content_hash=null`. Publish hash binding once
it becomes known. Do not invent a hash before the host has the corresponding tokens.

## Exact observation surfaces

| Event | Payload beyond the common envelope | Why expose it? | Policies supported |
|---|---|---|---|
| `request_arrived` | Task category, online-known prompt length, ordered full-block prefix hashes, optional online-known session identity. | Establish current request context, without revealing a later turn. | Request-aware policies; task-aware classification. |
| `request_admitted` | Request/sequence IDs, admission ordinal, current queued/active status, ordered known prompt-prefix hashes. | Reproduce the online helper's actual `_add_processed_request` observation boundary after scheduler admission. | OPT-Online, UniCache-OPT. |
| `request_state` | Waiting/running/preempted/finished transition and its observed time. | Maintain a current request view; observe interruption/recompute boundaries. | Online lookahead and request-aware policies. |
| `lookup` | Lookup ID, content hash, prefix depth, hit/miss, resident block reference if present, computed/usable flags. | Observe the real cache lookup before any allocation it causes; distinguish resident from usable KV. | ARC/LeCaR histories, reuse predictors. |
| `lookup_completed` | Lookup ID, allocation result/block reference, content hash and outcome. | Mark the native post-allocation notification boundary. Reference ARC/LeCaR apply their `miss` logic here to match current behavior. | Exact ARC/LeCaR ports. |
| `block_created` | New block reference, owning request/sequence when known, parent prefix identity, prefix depth, valid-token count, content hash or null, computed flag. | Track allocation lifetime and prefix structure without inheriting a baseline tracker. | Every policy; depth ranking for OFFSET and tie-breaks. |
| `block_changed` | Hash binding, valid-token count, prefix depth, parent identity and computed-state changes; old/new block references for copy-on-write or transfer. | Handle mutable-to-immutable transitions, physical-slot reuse and copying correctly. | Every policy's identity/state bookkeeping. |
| `sequence_step` | Engine iteration, prefill/decode stage, ordered block references, prompt length, generated length so far, task category and per-block decode/prefill classification. | Reconstruct actual scheduler access callbacks, including active blocks. This is simulated step-use, not a hardware memory-access trace. | LRU, LFU, FIFO, ARC, LeCaR, WA, UniCache. |
| `sequence_release` | Final ordered block references, release time, last step's time/stage/lengths/category, and reason: completion, preemption or other release. | Reproduce deferred sequence-level timestamp/count updates at the same boundary as native policies. | Exact LRU/ARC/LeCaR/UniCache bookkeeping. |
| `block_access` | Block reference, access reason, step stage, block decode flag and task category. | Capture direct block-access observations not represented by a sequence step. Emit only at explicitly audited sites; define overlap with `sequence_step` to prevent double counting. | Recency, frequency, reuse intervals, classification. |
| `eligibility_changed` | Block reference, new reference count, computed/immutable flags, eligibility and transition reason. | Supply legal candidates, track reuse-to-active and release-to-idle transitions. | Every eviction policy. |
| `block_evicted` | Evicted block reference/content identity and the causing decision ID. | Record the actual outcome and maintain ghost histories. | Every policy; especially ARC/LeCaR. |
| `prefill_accounted` | Request ID, task category, cached prompt tokens, total prompt tokens and accounting ordinal. | Reproduce first-recorded hit accounting, per-task hit windows and adaptive efficiency weighting. One event per native first accounting record. | UniCache; general online learning. |
| `engine_iteration` | Start/end boundary, iteration index, actual prefill counter before/after, ordered prefill request IDs. | Reproduce the online helper's progress index. This counter is **not** the same as the first-accounted-request ordinal. | OPT-Online, UniCache-OPT; periodic updates. |
| `request_finished` | Request/sequence IDs, completion/cancellation status and observed time. | Retire current request state. | Request-aware and online-lookahead policies. |

Avoid duplicate information becoming duplicate accesses: document which event is the
source of truth for each observation. Emit events unconditionally for the v2 route,
not based on which reference policy is being compared.

Task labels and decode classification are public request/block covariates. Expose an
online-known `workload_class` (for example `apps`, `alfred`, or `sharegpt_text`) and
the native accounting category as separate fields, identically for every policy.
Do not collapse them into one label: native WA treats APPS as single-turn, whereas
UniCache's block classifier distinguishes single-turn-random; accounting classification
can also differ from block classification. Each reference port implements its own
mapping from these public covariates. Document the decode-block rule, including partial
last blocks, separately from the scheduler's prefill/decode stage. No future session
length is needed to classify an already observed request.

Do not pass a host-defined retention score, task weight, policy queue name,
policy-specific reset, universal access count, or reuse distribution. Policies derive
those from the ordered observations. A derived inspection snapshot may be added later,
but its definitions must be independent of any baseline.

### Decision input/output on the worker channel

```json
{
  "type": "evict",
  "api_version": 2,
  "decision_id": 17,
  "through_event_id": 408,
  "cache_id": "gpu:0",
  "now": 120.5,
  "reason": "allocation",
  "events": [],
  "eligible_count": 412
}
```

The worker applies every queued event through ID 408 before invoking `evict`:

```json
{
  "decision_id": 17,
  "victim": {"block_id": 42, "generation": 3}
}
```

An event-only batch returns its last acknowledged event ID. Do not transmit the
entire cache on every eviction; maintain eligibility from events, with an optional
read-only candidate snapshot for diagnostics/resynchronization. Never evict before
relevant pending events have been acknowledged. Repeated eviction requests require
an outcome event between decisions so the victim cannot be returned twice.

The host rejects stale generations, ineligible/active blocks, malformed results,
wrong-cache responses, timeouts and crashes. There is no silent fallback.

## What remains fixed in the host

Physical allocation/freeing, reference counts, prefix matching, token/content hashes,
computed-block validity, copy-on-write, request scheduling, cache budgets, and trusted
score calculation remain in the simulator. Agent code cannot change their rules.

The only normal action is choosing an eligible victim. All retention logic and its
bookkeeping live inside the submitted policy. When no eviction is necessary, the
policy still receives observations; it cannot manufacture requests or cache hits.

NONE is a configuration exception, not an eviction algorithm. To express it, validate
`CACHE_MODE` once in the isolated worker **before engine construction**. `disabled`
selects the existing no-prefix-caching path; capacities and other settings remain
fixed and the run is labelled a cache-disabled control. Reject mode changes during
replay. Do not pretend that evicting idle blocks immediately exactly reproduces NONE,
because active blocks can otherwise still participate in prefix reuse.

Offline OPT stays host-only. Even the online lookahead variants see only arrived and
admitted requests. Preserve helper observation order and index semantics when porting;
any bug fix or improved definition of OPT-Online is labelled separately from parity.

## Numerical/runtime support

### What each reference policy would implement itself

| Policy | State and rules inside `policy.py` | Required public observations |
|---|---|---|
| LRU | Last-access timestamps, deferred release updates, depth tie-breaks and heap maintenance. | Sequence steps/releases, lifecycle and prefix depth. |
| LFU | Prefill access counts and their native reset rules; depth tie-breaks. | Step stage, block membership, release/reuse transitions. |
| FIFO | First-access timestamps and native retention/reset behavior. | Step times and block lifecycle. |
| OFFSET-HEAD / OFFSET-TAIL | Ascending/descending prefix depth with native recency ties. | Prefix depth and LRU observations. |
| ARC | T1/T2 resident queues, B1/B2 content-hash ghost histories, adaptive target and replacement rule. | Identity, eligibility, access, eviction outcomes and post-allocation miss notification. |
| LeCaR | Recency/frequency rankings, eviction histories, discounted penalties, learned weights and seeded random selection. | ARC observations, access-count facts, clock, capacity and seed. |
| WA | Per-block prefill reuse intervals, task-specific exponential fits, fit schedule and survival scores. | Sequence steps with stage/category, lifecycle/reset boundaries and time. A lookup hit alone is insufficient: native WA samples prefill block observations. |
| UniCache | Task/decode mapping, priority queues, recency/count state, lognormal/offset/decay scores, per-task hit windows and adaptive weights. | Step/release metadata, prefix depth, lifecycle, workload class, raw accounting observations and time. |
| OPT-Online | Known-request prefix-use lists and progress counters, preserving native helper semantics. | Arrived/admitted prefixes and actual prefill-counter boundaries. |
| UniCache-OPT | UniCache state plus the same online helper. | Union of the preceding two rows. |

This includes UniCache's fitted constants, rolling-window sizes, decay factors,
weight-update schedule, clipping, stale-heap cleanup and tie-breaking in editable
policy code. None remains an invisible host-side advantage. Ghost histories retain
content identity after physical blocks are evicted, not just recycled block IDs.

Keep `policy.py` isolated and offline. Permit vetted, read-only numerical dependencies
with pinned versions, without mounting vLLM or the private reference-policy package.
Exact reference parity may require NumPy's RNG/float32 behavior for LeCaR and matching
numerical operations for WA/UniCache. Standard-library `random` is not automatically
an equivalent substitute for NumPy's generator. Provide dependency support without
providing a baseline implementation or a policy-specific parameter table.

Keep current runtime limits, aggregate-only tune submission feedback, rationale,
local smoke tests and custom-test launcher. Runtime events are available to the
policy process, not automatically written into the coding LLM's context. Disable
raw event dumps by default; private diagnostic traces are opt-in. Baseline/eval results
and the offline map remain outside the coding workspace and local-test sandbox.

## Proposed implementations through this one interface

These are implementation sketches, not runnable submissions or completed parity ports.
Every implementation below exports exactly the same `Policy(config)`,
`on_event(event)` and `evict(context)` methods. Changing policies replaces `policy.py`;
it does not change the host's event selection, tracker type, scheduler or permissions.
`--start-policy` selects editable source, not a hidden backing algorithm. An agent
starting with LRU can replace the entire class, add helpers and change every rule.

### Common plumbing, with no built-in strategy

Each policy keeps a facts table keyed by `(cache_id, block_id, generation)`, plus its
own algorithm state. A small, inspectable lifecycle helper may maintain immutable
metadata and eligibility from events. It must not calculate recency, frequency,
scores, queue membership, learned weights or ghost histories. Those remain editable.

```python
class Policy:
    CACHE_MODE = "prefix"

    def __init__(self, config):
        self.config = config
        self.facts = {}      # Identity, hash, depth, computed state, eligibility.
        self.state = {}      # Entirely policy-defined; includes active-block state.
        self.pending = {}    # Decisions awaiting host outcome events.

    def on_event(self, event):
        self.apply_lifecycle_fact(event)  # Only shared physical facts.
        self.observe(event)               # Editable bookkeeping, described below.

    def evict(self, context):
        victim = self.choose(context)     # Editable algorithm, described below.
        self.pending[context["decision_id"]] = victim
        return {"block_id": victim.block_id, "generation": victim.generation}
```

Here `apply_lifecycle_fact`, `observe` and `choose` would be
ordinary code shipped in `policy.py` or a documented, inspectable neutral helper.
They are notation for the sketches below, not imports from a native evictor.
`block_evicted` confirms the pending decision before removing membership or adding
ghosts. Eligibility excludes active blocks from decisions but does not erase their
access history. Host rejection fails the submission; it cannot trigger a baseline.

### LRU, LFU, FIFO and the two OFFSET policies

Their `choose` method pops the smallest valid key from a policy-owned heap. The
following Python tuples specify each ranking, with block ID breaking remaining ties.
`depth` means cumulative prefix tokens, not the tokens contained in one block.

| Policy | Editable `observe` bookkeeping | Heap key, smallest evicted first |
|---|---|---|
| LRU | Remember the latest sequence-step time; apply the native deferred final-block update at `sequence_release`. Handle direct access sites separately. | `(last_accessed, -depth, block_id)` |
| LFU | Count the native prefill block observations, preserving counts across the same reuse transitions and resetting on content replacement. Do not count every event or every decode step. | `(access_count, -depth, block_id)` |
| FIFO | Set `first_accessed` on the first applicable native access; do not refresh it on a hit. Reproduce tracker lifetime/reset boundaries. | `(first_accessed, -depth, block_id)` |
| OFFSET-HEAD | Reconstruct LRU's access timestamps; rank shallow prefixes first. | `(depth, last_accessed, block_id)` |
| OFFSET-TAIL | Reconstruct the same timestamps; rank deep prefixes first. | `(-depth, last_accessed, block_id)` |

These are ordering rules, not a substitute for porting the tracker. For exact native
parity, also preserve insertion/update timing, lazy invalidation and heap cleanup.
For example, native `update` can modify metadata without immediately inserting a new
heap entry. A fresh `min()` over all current metadata need not match that behavior.
Corrected variants may use cleaner bookkeeping, but must have separate result labels.

### ARC

State: eligible resident heaps `T1` and `T2`, content-hash ghost histories `B1` and
`B2`, adaptive target `p`, native access counters and the previous `miss_in_B2` flag.
All state is inside the submitted implementation.

```python
# observe(event), abbreviated:
# - Derive deferred recency/count updates from step + release facts.
# - When becoming eligible, insert in T1 if count == 1, otherwise T2.
# - On lookup_completed(miss), apply the native ghost adaptation:
if missed_hash in B1:
    p = min(capacity, p + max(1, len(B2) // len(B1)))
    del B1[missed_hash]
    miss_in_B2 = False
elif missed_hash in B2:
    p = max(0, p - max(1, len(B1) // len(B2)))
    del B2[missed_hash]
    miss_in_B2 = True
else:
    miss_in_B2 = False

# choose(context), replacement branch:
if not T1:
    victim = oldest_valid(T2)
elif not T2:
    victim = oldest_valid(T1)
elif len(T1) > p or (miss_in_B2 and len(T1) == p):
    victim = oldest_valid(T1)
else:
    victim = oldest_valid(T2)
# On confirmed eviction, put its content hash in the corresponding B1/B2.
```

The full port must include native `evict`'s capacity/ghost-trimming branches before
replacement, empty-queue fallbacks, tie ordering and lazy-heap lengths. The snippet
shows the adaptive rule, not the complete ARC implementation. In this repository,
`T1/T2` hold evictable blocks; claiming parity with textbook ARC would be a different
claim. A general ARC implementation can also be written under this same interface.

### LeCaR

State: an LRU heap, an LFU heap, two hash-keyed eviction histories with eviction times,
weights `W=[0.5, 0.5]`, and a seeded NumPy RNG. Reconstruct native deferred recency and
count changes, then update both rankings when blocks enter the eligible set.

```python
# choose(context):
lru = oldest_valid(LRU)
lfu = least_frequent_valid(LFU)
if lru == lfu:
    victim, expert = lru, None
else:
    expert = 0 if rng.rand() < W[0] else 1
    victim = [lru, lfu][expert]
# On confirmed eviction, remember hash/time in that expert's bounded history.

# observe(lookup_completed(miss)) when a hash reappears in an expert's history:
reward = [0.0, 0.0]
reward[expert] = -(discount ** (now - eviction_time))
W *= numpy.exp(learning_rate * numpy.array(reward, dtype=numpy.float32))
W /= W.sum()
# Apply the native 0.01/0.99 weight limits; remove the matched ghost entry.
```

Private reference constants: RNG seed `123`, learning rate `0.45`, history capacity
equal to cache capacity, `discount = 0.005 ** (1 / capacity)`, NumPy float32 weights.
Handle empty heaps and agreement exactly as native code: when experts agree, native
code adds no blame history. These constants are editable policy code, not environment
requirements; a rediscovering agent may choose other values.

### WA

State: per-block last-access and last-prefill-observation times, task mapping, one
recency heap per task, collected reuse intervals, fitted rates and last-fit time.
Collect intervals at the native prefill observation sites, including zero intervals
where native code does; preserve the tracker's reuse/reset rules.

```python
# On the same scoring boundary as native code, fit if no successful fit exists
# or the configured fit interval has elapsed:
for task, intervals in samples.items():
    if len(intervals) >= 5:
        _, scale = scipy.stats.expon.fit(intervals, floc=0)
        rate[task] = 1 / scale
        intervals.clear()
# Advance last_fit_time only when at least one fit succeeds.

# choose(context): one oldest valid candidate per task, then compare scores.
candidates = [oldest_valid(q) for q in task_heaps if has_valid(q)]
score = lambda b: 0 if b.task not in rate else numpy.exp(
    -rate[b.task] * (context["now"] - b.last_accessed))
victim = smallest_score_then_deepest(candidates, score)
```

The configured fit interval is public environment context. Preserve native candidate
queue order for remaining ties and return unchosen candidates to their heaps. On a
single task, candidate selection can reduce WA to recency ordering; a fitted model
alone does not ensure different choices.

### UniCache (`SimpleHYBRID`)

State: block category/decode/count/recency metadata; seven heaps (`evict_first`,
`single`, `text`, `file`, `image`, `search`, `agentic`); per-task hit windows; adaptive
weights `alpha`; and the last weight-update accounting ordinal. All are agent-owned.

1. Derive native block categories from public workload context and per-block decode
   facts. Put single-turn decode blocks and single-turn-random blocks in `evict_first`.
2. Reconstruct deferred last-access updates and prefill-only count changes. On becoming
   eligible, insert into the appropriate heap using the ordering below.
3. On `prefill_accounted`, append hit tokens to that task's 500-entry window and decay
   other task windows by `0.95`, matching native accounting/category mapping.
4. On the native eviction scoring boundary, update task weights once when the
   accounting ordinal is divisible by 10. Do not substitute engine iteration count.

| Heap | Native ordering, smallest key first |
|---|---|
| `evict_first` | `(-depth, last_accessed, block_id)` |
| Each multi-turn task | `(last_accessed, -depth, block_id)` |
| `single` | `(-depth, access_count, last_accessed, block_id)` |

```python
# choose(context), abbreviated:
if has_valid(evict_first):
    return pop_valid(evict_first)
candidates = [pop_valid(q) for q in decay_queues if has_valid(q)]
maybe_update_alpha_at_native_boundary()
if len(candidates) == 1:
    return candidates[0]

def retention_score(b):
    age = context["now"] - b.last_accessed
    if b.is_multi_turn:
        mu, sigma = priors[b.task]
        return alpha[b.task] * (1 - lognormal_cdf(int(age), mu, sigma))
    offset = max(0, min(1, 1 - b.depth / 2048))
    return alpha["single"] * offset * native_age_decay(age)

victim = native_lowest_score_then_deepest(candidates, retention_score)
# Return unchosen candidates to their heaps; commit removal on block_evicted.
```

Private reference priors `(mu, sigma)`: text `(4.77, 1.279)`, file `(5.07658, 1.06383)`,
image `(4.4449, 1.09066)`, search `(4.961663, 1.13877)`, agentic `(1.81, 1.092)`.
Native single-turn decay uses the integer-age precomputed `exp(-age / 600)` table
below 20,000 seconds and a continuous-age exponential beyond it. Preserve its exact
score comparison/tie tolerance and any configured ablation behavior in reference tests.

The editable `update_alpha` reproduces: hit share divided by queue capacity share
with `eps=0.001`; normalize positive efficiencies by their mean; raise to power
`1 / 0.2`; combine `0.1 * old + 0.9 * scaled`; clip using native mean ± two standard
deviations with bounds `0.001` and `10`. Keep unchanged weights for invalid categories.
Native capacity share uses heap lengths, including lazy entries, divided by eligible
table length. Preserve this for parity; evaluate corrected occupancy separately.
Also preserve the native heap cleanup and classification-update behavior.

This exposes enough to write UniCache's full mechanism, rather than merely changing
its final victim. The numerical priors need not be supplied to an LRU-start agent:
it has online observations from which it can fit its own reuse model. Expressibility
does not imply it will infer these exact fitted constants within 30 submissions.

### OPT-Online

State: prefix-hash lists of admitted-request ordinals, an admission counter and the
native prefill-progress counter. Build these only from requests already admitted.

```python
# observe(request_admitted): append this admission ordinal for each prompt prefix.
# observe(engine_iteration): advance the documented native prefill counter.
def next_known_use(content_hash):
    uses = admitted_uses[content_hash]
    while uses and uses[0] <= prefill_counter:
        uses.pop(0)
    return uses[0] if uses else float("inf")

# When a block becomes eligible, snapshot next_known_use into its heap key:
key = (-next_known_use(b.content_hash), b.last_accessed, -b.depth, b.block_id)
# choose(context): pop the smallest valid key, as native OPTEvictor does.
```

Preserve refresh/cleanup timing: native code does not recompute all priorities on
every new admission. Audit the helper's mixed hash key representations and distinct
admission/prefill index semantics during the port. Any compatibility behavior belongs
in editable reference code, with fixes labelled separately. No unarrived request is
provided, so this policy cannot reconstruct offline OPT's full future-use oracle.

### UniCache-OPT (`SimpleHYBRID-OPT`)

Use the complete UniCache state and event handlers plus OPT-Online's admitted-prefix
index. On eligible insertion, snapshot `next_known_use`. Prepend `-next_use` to each
UniCache heap key. Continue servicing `evict_first` before the other queues; among
their candidates prefer larger next-use, then lower UniCache score, then deeper
prefix using native comparisons. Preserve its single-candidate shortcut and refresh
timing. This is an online hybrid in this repository, despite the `OPT` suffix.

### NONE and offline OPT

NONE uses the same class contract with `CACHE_MODE = "disabled"` at initialization;
it receives no prefix-eviction decisions. It is a disabled-cache control, not a policy
the agent must emulate by choosing bad victims. Record the mode in results.

Offline OPT remains a host-only reference using its future-use map. Its information
advantage cannot be made attainable under an online-only contract. All other listed
baselines must pass the same v2 reference-port gate before we call them attainable.

### What establishes attainability

For every online baseline, execute a self-contained private `policy.py` through the
**same isolated worker and public messages** as an LLM submission. Require the parity
checks below, including pressured caches, mixed tasks, active blocks and ghost hits.
Record source hashes, dependency versions and replay commands. Meet the agent's CPU,
memory and callback limits; optimize batching/indexes or revise the shared limits if
a reference port cannot finish. No privileged native imports or special timeouts.

Reference implementations are validation artifacts, not automatically visible to the
coding agent. Selecting one as the starting policy intentionally supplies its source.
Starting from LRU supplies LRU only. This keeps both claims testable: the interface
permits every online baseline, and the agent may independently discover a better rule.

## Improvement opportunities beyond existing UniCache

### What the completed experiments actually show

These are token-weighted **eval** hit ratios from the completed 1B budget study,
not predictions for v2. Values are percentages; gaps are percentage points (pp).
The best observed online column is descriptive across the measured online references;
it is not an additional policy selected on an unseen holdout.

| Workload | GPU blocks | UniCache | Best observed online | Offline OPT | OPT minus UniCache |
|---|---:|---:|---|---:|---:|
| ShareGPT | 16,384 | 61.86% | LeCaR, 62.49% | 81.81% | 19.95 pp |
| MASH-QA | 98,304 | 52.22% | UniCache, 52.22% | 58.58% | 6.37 pp |
| MASH-QA | 32,768 | 23.75% | UniCache, 23.75% | 49.99% | 26.24 pp |
| ALFRED | 32,768 | 54.00% | LeCaR, 64.79% | 83.09% | 29.10 pp |
| CodeParrot | 32,768 | 55.45% | UniCache, 55.45% | 68.21% | 12.76 pp |
| CodeParrot | 50,000 | 68.03% | UniCache, 68.03% | 68.21% | 0.18 pp |

Sources: [primary scores](result/cache-budget-study/2026-11-01/baseline_eval.csv),
[32,768-block counterpoints](result/cache-budget-study/2026-11-01/counterpoint_baselines.csv),
[reproduction report](result/cache-budget-study/2026-11-01/report.md).

ALFRED demonstrates an existing online method beating UniCache by **10.79 pp**;
ShareGPT shows a smaller **0.63 pp** advantage. CodeParrot at 32,768 blocks provides
more oracle separation than at 50,000, but no tested online policy beats UniCache
there. An offline gap is a search opportunity, not a guaranteed achievable online gain.
These are exploratory results from the existing split/seed, not confidence intervals.

### Concrete changes the expanded interface would allow

Each row is a hypothesis to test against the unchanged native reference. None is an
already demonstrated improvement unless stated above.

| Area in current UniCache | Editable improvement | Why it could help / limitation |
|---|---|---|
| Only one candidate per task, ordered mainly by recency for multi-turn tasks | Add frequency, ARC-style ghosts or a LeCaR-style recency/frequency mixture within each task; compare several candidates. | Homogeneous task runs bypass cross-task score comparison. ALFRED's LeCaR result is concrete motivation for richer within-task choices. |
| Fixed task-level lognormal priors | Fit online reuse distributions; condition on recent reuse, session and prefix-depth features, with fallback priors. | Adapt to workloads whose reuse differs from the fitted constants. Track censored/non-reused entries too; fitting only observed hits can bias estimates. |
| Survival score `1 - F(age)` | Test probability of reuse within horizon `h`, conditional on no reuse yet: `(F(age+h) - F(age)) / (1-F(age))`, with numerical guards and a non-reuse component. | Distinguishes residual reuse likelihood from unconditional survival. Must calibrate using observed data; no guarantee of better hit ratio. |
| Fixed 2,048-token single-turn depth scale | Learn depth/usefulness curves or normalize by observed prompt lengths. | Current clipping gives all sufficiently deep single-turn candidates zero offset value. Longer prefixes may still be valuable. |
| Task labels mapped from workload/request identifiers | Learn finer groups from online request and reuse features; merge categories with sparse evidence. | Dataset names are coarse proxies for behavior. Keep future session lengths and later turns hidden. |
| Weight capacity uses heap lengths with stale entries | Compare unique eligible block counts and actual resident-block occupancy, with explicit denominators. | Decouple resource estimates from lazy-heap artifacts. This is a separately labelled correction, not silent baseline parity. |
| Fixed 500-entry windows, inactive-task decay, update period and temperature | Tune smoothing or use drift-sensitive windows and uncertainty-aware weights. | Could adapt faster or avoid unstable cross-task allocation. Needs mixed workloads; single-task tests do not exercise this mechanism well. |
| Hard priority for single-turn decode/random blocks | Use observed reuse evidence to soften the rule for repeatedly shared content. | Some predicted-low-value classes may have useful repeated prefixes. Remains victim selection; bypass/admission control would require a separate action extension. |
| Known queued requests are used only by the separate hybrid | Add online queued-prefix demand to a learned retention score. | v2 exposes this information equally to all policies. It provides no information about requests that have not arrived. |
| Implementation overhead and lazy queues | Bound history/heap growth, reduce redundant updates, maintain efficient indexes. | May improve policy runtime and feasible search complexity; it does not by itself establish improved hit ratio or serving latency. |

The first experiment should pair unchanged UniCache with an editable faithful v2 port,
then try a within-task recency/frequency hybrid on ALFRED and CodeParrot at 32,768
blocks. Change one mechanism at a time, use tune for selection and keep eval feedback
outside the agent. Measure overhead alongside hit ratio. Repeated coding runs/seeds
and a fresh final holdout are needed before generalizing beyond exploratory results.

Recompute-cost-aware eviction is another possible objective, but hit ratio alone does
not establish a latency improvement. It would require a validated timing/cost provider
and separate latency reporting; the proposed v2 interface does not create that data.

## Implementation sequence

1. **Audit and pin semantics.** Enumerate native call sites, task classification,
   deferred updates, miss ordering, reference-count transitions, hash binding and both
   prefill counters. Capture small private native traces before changing routing.
2. **Define typed v2 schemas.** Implement config, event and decision validation with
   serialization tests. Document exact ordering, access-count derivations and generation
   lifetime. No hidden policy-specific fields.
3. **Instrument shared mechanisms.** Add the event stream to allocator, block manager,
   scheduler, engine and output accounting. Initially run it in shadow mode; native
   decisions and all scores must remain unchanged.
4. **Implement the independent v2 evictor.** Its membership/eligible count comes from
   trusted lifecycle state, not a hidden baseline. It must satisfy the allocator's
   existing interface without inheriting policy-specific tracking behavior. Forward
   global/request events to the applicable per-cache workers in consistent order.
5. **Create private reference ports.** Implement each online baseline entirely through
   v2, with its native constants and dependency versions. No privileged helper imports.
   Port actual bookkeeping, reset rules, queue maintenance, RNG and tie-break behavior,
   including implementation quirks; record defects separately rather than silently fixing.
6. **Run parity and isolation gates.** Compare native and v2 behavior as specified below.
   Do not label v2 sufficient for all policies until the entire reference matrix passes.
7. **Update the agent harness.** Add `--policy-interface v1|v2`, retain v1 for old results,
   initialize actual editable starting code, update API docs/prompt/smoke tests, and
   record interface/event/dependency versions in every manifest.
8. **Run paired agent trials.** Begin with ALFRED and CodeParrot at 32,768 blocks, LRU
   start and identical traces/budgets/submission caps. Reference implementations remain
   private. Compare v1/v2 on repeated coding runs; distinguish exploratory existing eval
   results from any fresh holdout used for final claims.

## Required verification

- Shadow instrumentation: native policies produce the same decisions, scores and
  completed requests with observation enabled/disabled.
- Exact event-order tests: lookup hit/miss and post-allocation notification, reuse,
  active-to-idle transitions, generation reuse, mutable hash binding, copy-on-write,
  sequence release, cancellation/preemption and equal timestamps.
- Numerical tests: deterministic RNG, decay, reuse fits, float precision, queue ties
  and periodic weight-update boundaries match the private reference configuration.
- Full native-versus-v2 parity for every covered online policy: identical eviction
  sequence where decisions occur, final hit-token counts, prompt totals, completion
  ledger and online counters. A matching score without exercised evictions is insufficient.
- Use synthetic cases designed to exercise ghost hits, task mixing, single-turn decode,
  frequency/recency conflicts and online queued reuse. The five homogeneous workloads
  alone do not exercise all UniCache branches.
- Include mixed-task validation inputs generated deterministically from existing
  sessions, clearly labelled new tests rather than the paper's unreleased mixed traces.
- Validate at both saturated and pressure-inducing cache budgets, including 32,768
  blocks where existing ALFRED/CodeParrot results show headroom. Exercise multiple
  seeded traces and candidate batching sizes; event batching must not alter decisions.
- Test future/eval/map denial, source isolation, disallowed dependencies, invalid
  victim generations and attempted score manipulation. No runtime agent state survives
  between independent tune/eval or repeated evaluations.
- Check NONE's disabled-cache mode separately. Continue validating offline OPT only as
  a private benchmark; no oracle information is added to v2.

## Completion criteria and deliverables

Deliver versioned schemas, a documented event contract, shared event emitters, an
independent evictor/worker, actual editable starting policies, private online reference
ports, parity/isolation results, updated sandbox tests and reproducible paired-run
commands. Preserve the current summary/report tooling with interface version recorded
so incompatible experiments cannot be silently combined.

The claim on completion is: **the agent's interface can implement all listed online
baselines from scratch without additional privileged observations**. It is not a claim
that an agent will independently rediscover the same algorithms or constants.
