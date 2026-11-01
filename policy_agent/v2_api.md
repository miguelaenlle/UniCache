# Editable policy interface v2

Implement `Policy(config)`, `on_event(event)`, and `evict(context)` in policy.py.
All policy queues, counters, histories, models and weights are yours to change.
There is no backing baseline and no fallback decision. The starting implementation
contains its algorithm and bookkeeping as editable source; other baseline sources,
scores, eval inputs and offline OPT maps are not available in this workspace.

`config` supplies cache_id, capacity in blocks, block_size in tokens, simulated clock
units, API/event versions and numerical dependency versions. It also supplies
serving_model, model/KV dimensions, kv_dtype, kv_bytes_per_block and scheduler limits
(max_model_len, max_num_seqs, max_num_batched_tokens, preemption_mode).
Time is virtual seconds.
State resets on every independent replay. Python standard library, NumPy, SciPy and
their vetted import dependency charset_normalizer are available. No simulator imports,
trace files, network, subprocesses or arbitrary filesystem writes are allowed in the
policy worker. Each callback batch has 30 seconds; each worker has 600 CPU seconds.

`CACHE_MODE` is an optional fixed class-level constant, either "prefix" (default)
or "disabled" for the NONE control. Startup inspects this declaration in isolation
without calling your constructor. The actual prefix-policy constructor runs once
complete model/scheduler context is available, even when the trace needs no evictions.

Events arrive in order, each with increasing event_id and time. Every pending event
is delivered before a decision. A block's identity is (cache_id, block_id, generation);
physical IDs are reused. Content hashes are opaque lossless strings. Depth is the
cumulative prefix token count. Active blocks are observable but cannot be evicted.

| kind | Main fields / meaning |
|---|---|
| block_allocated | block_id, generation; physical slot assigned to new content |
| block_created | block_id, generation, parent/content identity, token metadata |
| hash_binding | block_id, content_hash, depth; immutable prefix identity becomes known |
| block_activated | block_id, content_hash, reference count; cached content is reused |
| refcount_changed / block_released | block_id, reference count; physical lifecycle |
| block_computed / copy_on_write | Computation validity / physical copy observations |
| eligibility_changed | block_id, generation, eligible; when true also content_hash and depth |
| sequence_step | sequence_id, request_id, block_ids, now, is_prefill, workload_class, prompt_tokens_len, output_tokens_len (generated so far) |
| sequence_release | sequence_id, final block_ids, last_step; boundary for deferred access updates |
| lookup | content_hash, hit; lookup before replacement allocation |
| lookup_completed | content_hash, hit; after allocation, which may already have required eviction |
| prefill_accounted | category, hit_tokens, prompt_tokens, ordinal; one native first-prefill accounting record |
| request_admitted | request_id, ordinal, prefix_hashes from this admitted request only |
| engine_iteration | Native iteration/prefill progress counters; distinct from accounting ordinal |
| request_state / request_finished | Online scheduling/completion transitions |
| block_evicted | block_id, generation, content_hash, decision_id; confirmed decision outcome |

Not every event is an access. Choose your own definitions and reset rules from raw
facts; do not increment a frequency counter once for every event. A sequence release
can describe final access observations that native policies apply after the last step.
Keep metadata for active blocks if your algorithm needs it. Keep evicted hash identities
if you want ghost histories. Later turns and unseen requests are never provided.

`evict(context)` receives cache_id, now, decision_id, through_event_id, eligible_count,
and reason. Return exactly `{"block_id": 42, "generation": 3}` for an eligible block.
Do not return None. The host rejects active, stale, missing or malformed victims.
Trusted allocation, matching, scheduler behavior and score calculation are fixed.

Run `python smoke_test.py` for synthetic checks. You may write custom test scripts in
the workspace and execute `python run_local.py tests/example.py`. They are offline and
consume no submissions. Submit with `python submit.py --rationale-file rationale.md`.
Only aggregate tune feedback is returned. Stop with
`python submit.py --done --reason "finished"`.
