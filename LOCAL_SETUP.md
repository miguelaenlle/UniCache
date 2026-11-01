# Local simulator setup

Cloned from https://github.com/xsyslab/UniCache at
`bc51340e3d8a623ecdfc6372ed8b64ec978f1b15`.

Python 3.10 is installed in `.venv`, with PyTorch 2.6.0,
Transformers 4.53.2, Tokenizers 0.21.2, and SciPy 1.14.1.
The installed versions are recorded in `requirements/simulator-macos.lock.txt`.
Local configuration/tokenizer files are available for Llama-3.1-8B-Instruct
and Llama-3.2-1B-Instruct; the simulator does not need model weights or a GPU.
The setup script downloads the 8B metadata. For the 1B workload experiments:

```bash
.venv/bin/python scripts/local_sim.py download --model models/Llama-3.2-1B-Instruct
```

See `EXPERIMENTS.md` for the five-workload, eight-policy hit-ratio matrix.
Downloaded datasets, model files, and experiment outputs are kept locally
and are not included in Git.

## Run

From this repository directory:

```bash
./sim
```

No environment activation is required. This uses the included 20-session
ShareGPT sample, the UniCache policy (`SimpleHYBRID` in the source),
2,000 simulated GPU blocks, and a block size of 16 tokens.
The launcher uses vLLM V0 and virtual time and writes upstream outputs
under `result/local/`. It passes additional arguments to `run_simulation.py`;
for example, `./sim --eviction-policy LRU`. Comparison scripts additionally
write a `comparison.csv` and per-policy logs.

Compare the seven online policies:

```bash
bash scripts/compare-policies.sh
```

Sweep cache capacity and include the upstream offline OPT workflow:

```bash
bash scripts/compare-policies.sh --blocks 2000 5000 10000 --policies LRU LFU FIFO LeCaR ARC WA SimpleHYBRID OPT
```

OPT automatically runs a trace-collection pass for each capacity first.
This follows the upstream oracle workflow; it is not an independent proof
that its prefill ordering is identical across policies.
Each sweep saves commands, dataset/configuration checksums, dependency versions,
and individual summaries. `hit_ratio` is the aggregate hit-token/prompt-token
ratio, rather than the unweighted mean of per-request ratios.

## Paper reproduction

Paper: https://jxing.me/pdf/unicache-sigmetrics26.pdf

The included demo is a setup check, not a reproduction of the paper's figures.
The full tokenized datasets are absent from the release. The repository
provides preparation scripts and experimental YAML files, but does not
clearly identify an exact mapping of those YAMLs to the paper's three final
traces. Do not label the demo or arbitrary hybrid configurations as those traces.

Inspect the inputs needed by an existing hybrid configuration:

```bash
.venv/bin/python scripts/local_sim.py doctor --config TaskData/Hybrid/test1/tasks_config.yaml
```

It lists missing prepared data files before any experiment starts. Once
the inputs in that YAML are available:

```bash
bash scripts/compare-policies.sh --config TaskData/Hybrid/test1/tasks_config.yaml --blocks 50000 70000 90000 --policies LRU LFU FIFO LeCaR ARC WA SimpleHYBRID OPT
```

Those block counts are an example sweep, not verified conversions of the
paper's memory fractions. The paper evaluates three workload mixtures and cache budgets relative to
memory remaining after model loading; block-size settings vary across released scripts.
Exact reproduction requires the authors' final trace construction/configurations,
matching tokenizer/model metadata, timing parameters, and cache-block budgets.
Section 6.4/Table 5 latency measurements use real GPUs; the CPU simulator
does not reproduce those measured latency results.

Dataset preparation instructions are in:

- `TaskData/SharedGPT-multi-turn/readme.md` (download, clean, deduplicate).
- `TaskData/AgentBankData/readme.md` (AgentBank download and ALFRED scaling).
- `TaskData/Codeparrot/readme.md` (codeparrot/apps conversion).
- `TaskData/ToolBench/readme.md` (external Preble workload download).
- `TaskData/mashqa_data/readme.md` (external MASH-QA download).

Use `.venv/bin/python` for those scripts. Their `data.sh` files contain
model paths that must match your selected tokenizer. A direct conversion is:

```bash
VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .venv/bin/python prepare_session_data.py \
  --raw_data_path /absolute/path/to/conversations.json \
  --save_data_path /absolute/path/to/sessions.jsonl \
  --tokenizer_path models/Llama-3.1-8B-Instruct
```

All policies in a comparison must use the same prepared data and settings.
Upstream preparation uses some unseeded randomness and UUIDs, so independently
regenerated datasets are not guaranteed to match the authors' exact samples.

## Environment maintenance

To recreate this macOS environment and fetch its tokenizer files:

```bash
./setup-simulator.sh
```

The launcher sets cache directories within the repository. Simulation runs
use local model files with Hugging Face offline mode enabled.

Local compatibility changes are limited to allowing a kernel-free editable
install on macOS and importing PyTorch's compiler modules before vLLM installs
its Triton placeholder, plus permitting 8-token blocks in explicit virtual-time
simulation. Eviction and scheduling implementations are unchanged.
The source-only install may warn that native `vllm._C` kernels are absent;
it is intended for CPU trace simulation, not actual model inference.
