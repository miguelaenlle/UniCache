# ShareGPT preparation for UniCache

Uses the dataset and preprocessing sequence in the authors' `readme.md`.
The original cleaning, deduplication, and session-preparation scripts are unchanged.

Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`

Dataset revision: `192ab2185289094fc556ec8ce5ce1e8e587154ca`

Downloaded file: `ShareGPT_V3_unfiltered_cleaned_split.json`

From this directory, the preprocessing commands were:

```bash
../../.venv/bin/python clean_ShareGPT_V3.py
../../.venv/bin/python deduplicate.py
```

Then, from the repository root:

```bash
VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  .venv/bin/python prepare_session_data.py \
  --raw_data_path TaskData/SharedGPT-multi-turn/ShareGPT_cleaned_valid_deduplicated.json \
  --save_data_path TaskData/SharedGPT-multi-turn/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl \
  --tokenizer_path models/Llama-3.2-1B-Instruct
```

The tokenizer is the local `unsloth/Llama-3.2-1B-Instruct` metadata download,
matching the model family used in Figure 1. The authors' `data.sh` currently
selects Qwen; the command above invokes the same preparation script directly
with the Llama tokenizer and writes to the path used by their ShareGPT YAML.
The upstream fixed chat-template date and special-token handling are preserved.

Counts: 94,145 original conversations; 92,542 after cleaning; 57,519 after
deduplication. The original deduplication removes every conversation whose
first question appears more than once, rather than keeping one duplicate.

The `sharegpt-*.log` files record preprocessing output. The provenance JSON
records file checksums and validation counts after preparation completes.
This step prepares the full dataset; it does not run eviction experiments
or establish the cache capacity and timing settings needed for Figure 1.
