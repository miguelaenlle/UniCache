# Sim-to-real transfer of eviction policies: methodology

Status: design, 2026-11-25. This runs **before** the timing calibration in
`docs/gh200-calibration-plan.md`, which becomes Phase C. Advisor input:
`docs/reviews/gh200-plan-advisor-review.md`.

## 1. Question

**Q1 (hit ratio):** do the policy decisions the UniCache simulator makes hold on
a real GPU? If the simulator says policy A has a higher hit ratio than policy B
(an agent policy over LRU, or the best heuristic over the agent), does the
GPU agree, and by about the same amount?

**Q2 (latency):** do those hit-ratio differences become TTFT and ITL
differences on the GPU, and can the simulator predict them?

Why it matters: H1 scored every agent policy in simulation only, so its
headline numbers (+4.25 pp over LRU, −1.46 pp vs the best heuristic) are
simulator claims. H2 checked the simulator against a GPU only for heuristics,
on ≈100-request subsets. It found good agreement on four workloads and a
51 pp error on ShareGPT. Either answer is a result: if decisions transfer,
simulator-based policy search is justified; if they don't, a better simulator
is a prerequisite for the agent loop.

Hypotheses, written down before any GPU run:
* **Single-turn workloads (CodeParrot, MASH-QA) transfer.** Arrivals are fixed
  and no request depends on another's completion, so timing reaches the hit
  ratio only through which blocks are pinned by running requests. Expect
  |Δ_sim − Δ_GPU| ≤ 0.5 pp.
* **Multi-turn workloads (ALFRED, ShareGPT) may not transfer.** Follow-up turns
  arrive after the previous turn *finishes*, so the simulator's decode
  timing changes which blocks are still cached when they arrive. UniCache,
  LeCaR and the ALFRED agent also use constants measured in seconds, so a
  different time scale changes their behaviour.

## 2. Experimental design

### Constraint: replay is real time

A GPU replay runs at the trace's recorded arrival times, so a 1-hour trace
takes 1 hour of GH200 time. The H1 eval splits are 38–47 min each, so
replaying the H1 configurations directly would cost ≈ 20+ GPU-hours. We
therefore replay **short subsets (≤ 10 min each)** and choose each subset's
cache capacity so that it has **real KV-cache contention** (below). Sim and
GPU see exactly the same subset and capacity, so the comparison is
like-for-like. The trade-off is that we test the simulator's fidelity at
those configurations, not H1's exact numbers.

Arrivals are never compressed in Phase 1. Compressing them would change
concurrency and think times, which are part of what is under test.

### Hardware and serving setup

| Item | Choice | Reason |
|---|---|---|
| GPU | 1× NVIDIA GH200 (96 GB HBM3), NCSA DeltaAI, `ghx4` batch partition, 1 GPU per job | Available to us. Same H100 die, so aiconfigurator's `h100_sxm` tables are the closest existing timing model for Q2. 96 GB holds any block budget we choose. |
| Host | Grace (aarch64), bound to the CPU socket closest to the job's GPU (NUMA-local) | vLLM V0 with a 1B model spends a large share of each step on the CPU. Record `nvidia-smi -q` clocks and power, `nvidia-smi topo -m` and the hostname for every run. |
| Engine | The UniCache vLLM V0 fork in GPU mode (`--disable-simulation --device cuda`) | Same scheduler, block allocator and eviction hooks as the simulator, so any difference comes from execution and timing, not from different cache logic. |
| Model | Llama-3.2-1B-Instruct, bf16, dummy weights, outputs forced to the trace's recorded tokens | Same model as H1/H2. Forced outputs keep output lengths identical between sim and GPU. |
| Engine settings | Block size 8, `max_num_seqs` 32, chunked prefill off, prefix caching on, 0 CPU swap blocks, fixed seeds | Match H1. CPU blocks are only used for swap, which never happens with one sequence per request, so 0 is equivalent (checked once in sim). |

Ruled out: H200 nodes (hard to get), and running several replays on one GPU
(they would slow each other down and corrupt the timing).

### Traces and subsets

From each H1 workload we take the **eval split** (policies were selected on
tune, as in the report) and cut one subset:

| Workload | In? | Why | Subset rule |
|---|---|---|---|
| CodeParrot | yes | Largest agent gains in H1 (+4.9 to +11.4 pp); largest GPU policy gap in H2 (21.2 pp) | First 300 sessions by arrival (≈ 5 min at 1 req/s); single-turn |
| MASH-QA | yes | > 24 pp OPT−LRU headroom, yet the agent failed (≤ 1.3 pp); 6.8 pp GPU gap in H2 | First 300 sessions (≈ 5 min); single-turn |
| AgentBank ALFRED | yes | Multi-turn agentic: the timing-sensitive case; agent +10.1 pp at 8 GiB | First 60 sessions by arrival, each cut to its first 12 turns (mean session is 40 turns at 11 s think time, ≈ 8 min) |
| ShareGPT | yes | Multi-turn chat; H2's 51 pp error, attributed to decode timing | First 150 sessions (≈ 2.5 min of arrivals), at most 4 turns each (mean 2.9; think time ≈ 100 s) |
| AgentBank APPS | no | OPT−LRU headroom < 0.5 pp at every H1 capacity, so no contention to test | n/a |

The session counts and turn caps above are starting points. The final values
are fixed in Phase 0 so that each subset's simulated replay is ≤ 10 min. Every
cut keeps whole sessions, or whole turn prefixes of a session, and keeps the
recorded arrival times and seeded think times.

H2 is a warning here: its ALFRED subset (2 sessions) had no contention, and all
policies tied at 97.33%. A subset that is too small has nothing to test.
ALFRED therefore keeps many concurrent sessions.

### Ensuring real KV-cache contention

The block budget is set in the engine, so the GPU cache is as small as we say,
even though the GH200 has 96 GB. For each subset we sweep capacity in
simulation (256 to 32,768 blocks, in powers of 2; laptop, 0 GPU-hours) and
pick the capacity C\* that meets all of these:

1. **Headroom:** OPT − LRU ≥ 5 pp on the subset, the same threshold the report
   uses for H1.
2. **Real evictions:** the cache fills, and some evicted blocks are requested
   again later. Log the eviction count, the re-reference count and peak
   occupancy.
3. **Policies differ:** best minus worst heuristic ≥ 2 pp.
4. **No thrashing:** C\* ≥ 2× the largest request's blocks, and preemptions on
   ≤ 1% of requests, so contention comes from reuse rather than lack of room.

Among capacities that qualify, pick the one with the **largest OPT − LRU
headroom**. That criterion is policy-neutral, so it does not pick the capacity
where the agent looks best. If nothing qualifies, enlarge the subset (more
sessions) rather than shrinking the cache. On the GPU, condition 2 is checked
again from the run's own logs. A GPU run without evictions is reported as
invalid, not as agreement.

### Policies

* **Tier 1 (must run):** LRU (×3, the noise floor), UniCache, LeCaR, and the
  workload's best Table 2 agent policy (largest eval Agent − LRU):
  CodeParrot 8 GiB, MASH-QA 24 GiB, ALFRED 8 GiB, ShareGPT 4 GiB.
* **Tier 2 (if budget allows):** LFU, FIFO, and the other Table 2 agent
  policies for that workload. More policies make the rank correlation more
  meaningful.
* OPT runs in simulation only, as the headroom reference. Its future-use map
  doesn't hold under real timing.

Agent policies run at the subset's capacity, not the capacity they were tuned
at. The question is whether the simulator ranks them correctly at a given
configuration, and that holds at any capacity.

### Predictors (what "the simulator" means)

* **Phase 1, hit ratio:** the UniCache simulator exactly as H1 ran it (legacy
  timing: power-law TTFT, 10 ms per decode token), same code.
  Regression check: re-running one H1 configuration reproduces its hit ratio
  bit-for-bit after the code changes below.
* **Phase 2, latency:** the same GPU runs already log per-request TTFT
  (arrival → first token, including queueing), ITL and TPOT, so the GPU
  side needs no extra hours. Sim predictors: (P0) legacy timing; (P1)
  aiconfigurator `h100_sxm`, uncalibrated. Phase C adds the GH200-calibrated
  or native table.
* **Phase 2 load point (Tier 2):** at the recorded load, a 1B model leaves the
  GPU mostly idle, so latency gaps are small. One extra run set on CodeParrot
  at 4× arrival rate shows latency under load. That's valid for a single-turn
  trace with no think times, and the run takes 4× less wall time.

### Phase 0: prerequisites (laptop and login node, 0 GPU-hours)

1. **One clock for policies on the GPU.** The policy bridge reads the virtual
   clock, which never advances in GPU mode, while the core uses wall-clock
   time. So time-based agent policies would be broken on the GPU. Fix it and
   add a unit test.
2. **A Linux sandbox for agent policies.** The bridge hardcodes macOS
   `sandbox-exec`. Add a backend with networking off (`bwrap`, `unshare -rn`, or
   Apptainer `--network none`), and read each policy before copying it to
   DeltaAI.
3. **Subset builder and capacity sweep** scripts, eviction and occupancy
   logging, and a `loop_ms` column (whole-iteration wall time) in the step
   log for Phase 2.
4. **Build the fork on aarch64 (the main risk).** It needs torch 2.6 and must
   compile its CUDA kernels. Try the login node first; a CPU-only or GPU build
   job is the fallback (≤ 2 GPU-hours, budgeted separately).
5. **Pre-register:** the final subsets, C\*, the policy list, and the
   predicted direction of every Δ vs LRU, committed before the first GPU job.

### GPU jobs

* **Smoke (≤ 15 min):** the GPU is visible; an agent policy loads in the sandbox
  with networking off; the engine honours C\*; a 3-min CodeParrot window run
  twice (determinism); per-run wall time measured. Stop if anything is off.
* **One job per workload (≤ 1.5 h each):** Tier 1 runs in sequence. Each run
  writes a done-marker and is copied to `/work` as soon as it finishes, so a
  timeout loses at most one run.
* **Budget:** Tier 1 is 4 workloads × 6 runs × (≤ 10 min replay + ≈ 2 min
  startup), about **3–5 GPU-hours** plus the smoke job. Tier 2 adds about
  3 hours. Real run lengths from the smoke job set the final numbers.

## 3. Results format

Everything is written to `result/sim2real-transfer/<date>/` as CSVs plus a
`report.md` with these tables and one figure. Numbers are in pp, eval split
only.

**Table A: subsets.** Workload, sessions, requests, replay length (min),
capacity C\* (blocks and GiB), OPT − LRU (sim), best − worst heuristic (sim),
evictions (GPU), preemptions (GPU).

**Table B: hit-ratio fidelity** (like report Table 4). For each workload ×
policy: H_sim, H_GPU, |error|; per workload: mean and max error and the
GPU noise floor (spread of the 3 LRU runs).

**Table C: do decisions transfer?** For each workload × policy: Δ vs LRU in
sim and on GPU, the difference, and whether the sign matches. Rows for
Agent − LRU, Agent − Best and Best − LRU. Per workload: Spearman correlation
of the policy ranking. A decision **transfers** if the signs match and
|Δ_sim − Δ_GPU| ≤ max(0.5 pp, 2 × noise floor). Differences inside the noise
floor are reported as unresolved, not as agreement.

**Table D: latency (Phase 2).** For each workload × policy: TTFT mean and p90
and ITL mean, on GPU and under P0 and P1, with relative error. ΔTTFT vs LRU on
GPU, and whether its sign matches ΔH.

**Figure 1:** H_sim vs H_GPU scatter, one panel per workload, with a y = x
line and policies as markers.

**Summary paragraph** in report style: which hypotheses held, the
average Agent − LRU and Agent − Best on GPU vs sim, and what that means for
H1.

## 4. Threats to validity

* The "real" system is the same vLLM V0 control plane on a GPU, with dummy
  weights and forced outputs. It tests execution timing and scheduling, not
  model quality or a production V1 server.
* Subsets and C\* differ from H1's configurations, so H1's exact numbers are
  checked only indirectly.
* Agent policies run away from the capacity they were tuned for. That is fine
  for a fidelity test, but their absolute gains may differ from Table 2.
* Python policy code runs inside each step on the GPU, which slows
  completions. For multi-turn traces this is a real effect the simulator
  doesn't model. The per-step time of each policy vs LRU is recorded.
* GPU clocks can't be locked (no sudo), so clocks and power are recorded and
  the LRU repeats measure the resulting noise.
