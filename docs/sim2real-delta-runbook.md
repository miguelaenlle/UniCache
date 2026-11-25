# Running the sim2real GPU side on DeltaAI (one GH200)

Companion to [sim2real-transfer-methodology.md](sim2real-transfer-methodology.md)
and [sim2real-results.md](sim2real-results.md). Every GPU job requests **one
GH200** on the `ghx4` batch partition.

Lines marked **[check]** are assumptions about DeltaAI that step 1 confirms
before anything else runs.

## What runs where

| Where | What | GPU-hours |
|---|---|---:|
| Laptop | Simulator side (done), the bundle, analysis | 0 |
| DeltaAI login node | Checks, copying files, building the container | 0 |
| DeltaAI `ghx4`, 1 GH200 | Smoke job (≤ 20 min), then one job per trace (≤ 1.5 h each) | ≈ 3.5 |

Both sides use the same engine and the same script,
`scripts/sim2real/run_sim.py`. `--mode sim` runs the H1 simulator on the
laptop. `--mode gpu` passes `--disable-simulation --device cuda`, which makes
the engine run real forward passes on the GPU in real time. Both sides use
16-token blocks (H1 used 8, but this fork runs 8-token blocks on CUDA only
through xformers, which is x86-only).

## Step 0: log in and keep the session (once)

The tmux session `delta` holds an SSH connection with a control socket, so
copying files and running commands later need no second login:

```bash
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm/%r@%h:%p -o ControlPersist=12h \
    maenlle2@gh-login01.delta.ncsa.illinois.edu      # you type the password and Duo
```

## Step 1: check the allocation, quotas and tools (login node, 0 GPU-hours)

```bash
accounts                       # DeltaAI account name and GPU-hour balance  -> <ACCOUNT>
sinfo -s | grep -i ghx4        # partitions
quota                          # home / work / projects usage and file counts
ls -d /work/hdd/bcjw /work/nvme/bcjw /projects/bcjw   # [check] which are mounted on DeltaAI
which apptainer && apptainer --version
apptainer build --help | grep -q fakeroot && echo "fakeroot flag exists"
unshare --user --map-root-user --net true && echo "user+net namespaces OK"   # for the policy sandbox
nproc; free -g                 # login-node resources for the build
```

Stop and report to you if `accounts` shows no GH200 allocation, if the quotas
are near their limits, or if neither fakeroot builds nor `unshare` work.

## Step 2: make the work directory (login node)

```bash
export K=/work/hdd/bcjw/maenlle2/kvcal          # [check] the path from step 1
mkdir -p $K/{bundle,sif,results,logs,apptainer-cache,apptainer-tmp}
export APPTAINER_CACHEDIR=$K/apptainer-cache APPTAINER_TMPDIR=$K/apptainer-tmp   # keep home under quota
```

## Step 3: build and copy the bundle (laptop)

```bash
cd ~/Documents/Agentic-KV-Eviction/UniCache
.venv-sim312/bin/python scripts/sim2real/make_bundle.py alfred codeparrot@600 sharegpt@100x2 codeparrot@120
rsync -av -e "ssh -o ControlPath=~/.ssh/cm/%r@%h:%p" \
    result/sim2real-transfer/bundle/ maenlle2@gh-login01.delta.ncsa.illinois.edu:/work/hdd/bcjw/maenlle2/kvcal/bundle/
```

The bundle is about 45 MB and a few dozen files:
* the code at the committed HEAD;
* the Llama-3.2-1B config and tokenizer (no weights; the GPU runs dummy weights);
* the subset inputs;
* the 3 agent policies, with SHA256 sums;
* the container recipe and the job scripts.

I read every agent policy before it is copied.

## Step 4: build the container (login node; the main risk)

```bash
cd $K/bundle
MAX_JOBS=16 apptainer build --fakeroot $K/sif/unicache.sif unicache.def   # [check] fakeroot allowed
apptainer test $K/sif/unicache.sif                                        # torch + vllm import
```

The recipe:
* starts from a CUDA 12.6 Ubuntu 24.04 image (it has an arm64 variant);
* installs torch 2.6.0 from PyTorch's cu126 index ([check] that an aarch64
  wheel exists);
* compiles the fork's CUDA kernels for sm_90 only.

The compile likely takes 1–3 h.

If NCSA doesn't allow a heavy build on the login node, run the same command
in a 1-GPU batch job (≈ 1–2 GPU-hours). If torch 2.6.0 has no aarch64 CUDA
wheel, switch the base image to an NGC PyTorch container that ships torch 2.6.
Either way, I report back before spending GPU time.

Then a CPU-only check on the login node. The simulator inside the container
must reproduce a laptop number exactly:

```bash
apptainer exec --bind $K/bundle/subsets:/opt/unicache/TaskData/sim2real --bind $K/results:/results \
  $K/sif/unicache.sif bash -c "cd /opt/unicache && python scripts/sim2real/run_sim.py --mode sim \
  --block-size 16 --out /results/cpu-check --workloads codeparrot@120 --blocks 512 --policies LRU --jobs 1"
```

## Step 5: smoke job (1 GH200, ≤ 20 min)

```bash
cd $K && sbatch -A <ACCOUNT> bundle/smoke.sbatch
squeue -u $USER                     # wait for it
cat logs/s2r-smoke-<jobid>.out
```

The job checks five things:
* the GPU is visible;
* torch and vLLM import;
* a 2-min CodeParrot subset (LRU, 512 blocks) runs twice and gives the same
  hit ratio both times (determinism);
* the agent policy loads in the network-off sandbox;
* clocks and power are recorded before and after.

Pass criteria: both LRU runs finish with evictions > 0, they agree with each
other, and the agent run completes. I stop here and show you the result.

## Step 6: the measurement jobs (1 GH200 each)

These run only after you approve the smoke result. One job per trace, at the
pre-registered capacity:

```bash
cd $K
sbatch -A <ACCOUNT> --time=01:00:00 bundle/run_trace.sbatch alfred          <C_alfred>
sbatch -A <ACCOUNT> --time=01:30:00 bundle/run_trace.sbatch codeparrot@600  <C_codeparrot>
sbatch -A <ACCOUNT> --time=01:30:00 bundle/run_trace.sbatch sharegpt@100x2  <C_sharegpt>
```

* Each job runs LRU, UniCache, LeCaR and Agent, then two more LRU repeats.
  That is 6 real-time replays of about 5–10 min each.
* After each run, the job tars the results to `$K/results/gpu-<trace>.tar.gz`.
* Finished runs are skipped, so a resubmitted job resumes where it stopped.
* The three jobs can wait in the queue together. Each one uses one GPU.

## Step 7: copy results back and compare (laptop)

```bash
rsync -av -e "ssh -o ControlPath=~/.ssh/cm/%r@%h:%p" \
    "maenlle2@gh-login01.delta.ncsa.illinois.edu:/work/hdd/bcjw/maenlle2/kvcal/results/*.tar.gz" \
    result/sim2real-transfer/gpu/
```

Then fill the GPU columns of `docs/sim2real-results.md`:
* per-policy hit ratio;
* gap vs LRU;
* "transfers?", which needs the same sign and a gap within
  max(0.5 pp, 2 × the LRU noise);
* the evictions on each GPU run.

The simulator columns come from the block-16 pre-registration.

## Safety and resources

* No credentials are stored anywhere. You type the password and Duo. Later
  commands reuse the open SSH connection.
* Agent policies run with networking off: `unshare --net`, or the container
  flag `--network none`. No unsandboxed fallback exists.
* DeltaAI disk use stays under about 10 GB: the image (≈ 5–10 GB, one file),
  the bundle (≈ 45 MB) and the results (≈ 50 MB, tarred). Caches go to `$K`,
  not home.
* GPU: one GH200 per job. Hard `--time` limits. Nothing is submitted without
  your go-ahead.
