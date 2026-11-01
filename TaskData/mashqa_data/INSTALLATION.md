# MASH-QA preparation and comparison

Downloaded the three `*_webmd_squad_v2_full.json` files from the Google Drive
folder linked in the upstream `readme.md`. Original file IDs, sizes, and
checksums are recorded in `mashqa-download.json` and `mashqa-provenance.json`.

Ran the authors' `prepare_data.py` unchanged, producing 21,718 single-turn
requests from 5,574 source entries. The script samples the number of questions
per document using unseeded `random.gauss`. Its initial RNG state is saved in
`mashqa-random-state.json`; these samples are not claimed to be the exact
samples used in the paper.

Then ran the unchanged `prepare_session_data.py` from the repository root:

```bash
VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  .venv/bin/python prepare_session_data.py \
  --raw_data_path TaskData/mashqa_data/mashqa.json \
  --save_data_path TaskData/mashqa_data/mashqa_session_data.jsonl \
  --tokenizer_path models/Llama-3.2-1B-Instruct
```

This selects the same tokenizer as our ShareGPT comparison, instead of the
Qwen path currently hardcoded in upstream `data.sh`.

Run LRU, LFU, FIFO, and WA from the repository root:

```bash
bash TaskData/mashqa_data/run_figure1_mashqa.sh
```

Defaults follow the authors' MASH-QA script and YAML: 5,000 requests,
100,000 GPU blocks, 8 tokens per block, arrival rate 1, seed 42, max batch
sequences 32, and default prefill/decode timing. The upstream YAML calls
the task `Codeparrot`; it is preserved unchanged and points to MASH-QA data.

For the same 50,000-block capacity used in our ShareGPT test:

```bash
GPU_BLOCKS=50000 bash TaskData/mashqa_data/run_figure1_mashqa.sh
```

`SIMULATOR_TTFT` and `SIMULATOR_TPOT` override timing as in the ShareGPT script.
Outputs go to `result/figure1-mashqa/<timestamp>/comparison.csv`, plus per-policy
logs and summaries. Exact Figure 1 experiment settings remain unverified.
