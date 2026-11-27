# SPDX-License-Identifier: Apache-2.0
"""Per-step GPU latency for the virtual-time simulator.

The simulator decides *what* runs in each engine step (which requests, how many
of their prompt tokens were prefix-cache hits). This module answers *how long*
that step would take on a chosen GPU, using NVIDIA aiconfigurator's measured
kernel tables (``pip install aiconfigurator``).

    step_ms = scale[phase] * aiconfigurator_ms(batch) + overhead_ms[phase]

``scale`` and ``overhead_ms`` come from an optional calibration JSON that is
fitted against real GPU step times (see ``scripts/aic_timing``). Without one,
the raw aiconfigurator estimate is used (scale 1, overhead 0).

The legacy formula (``--simulator-ttft`` / ``--simulator-tpot``) is untouched
and remains the default; this model is used only with
``--simulator-timing aic``.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepBatch:
    """Token counts of one scheduled engine step, one entry per sequence.

    prefill_new_tokens[i]    prompt tokens computed this step (cache misses)
    prefill_cached_tokens[i] prompt tokens reused from the prefix cache
    decode_kv_tokens[i]      context length read by one decoding sequence
    """
    prefill_new_tokens: List[int] = field(default_factory=list)
    prefill_cached_tokens: List[int] = field(default_factory=list)
    decode_kv_tokens: List[int] = field(default_factory=list)

    @property
    def phase(self) -> str:
        return phase_of(len(self.prefill_new_tokens),
                        len(self.decode_kv_tokens))

    @classmethod
    def from_metadata(cls, seq_group_metadata_list, block_size: int) -> "StepBatch":
        """Build from the scheduler's ``SequenceGroupMetadata`` list.

        Cached prompt tokens use ``computed_block_nums`` -- the same quantity
        the simulator's hit-ratio statistics count -- so latency and hit ratio
        always agree on what was a hit. vLLM recomputes at least the last
        prompt token, so a fully cached prompt still costs one new token.
        """
        batch = cls()
        for meta in seq_group_metadata_list or []:
            for seq_data in meta.seq_data.values():
                if meta.is_prompt:
                    total = int(meta.token_chunk_size)
                    cached = len(meta.computed_block_nums or []) * block_size
                    cached = max(0, min(cached, total - 1))
                    batch.prefill_new_tokens.append(total - cached)
                    batch.prefill_cached_tokens.append(cached)
                else:
                    batch.decode_kv_tokens.append(int(seq_data.get_len()))
        return batch

    def to_fpm(self, wall_time_s: float = 0.0) -> dict:
        """aiconfigurator / Dynamo ``ForwardPassMetrics`` dict for this step."""
        def var(xs):
            if len(xs) < 2:
                return 0.0
            m = sum(xs) / len(xs)
            return sum((x - m)**2 for x in xs) / len(xs)

        return fpm_from_counts(**self.summary(),
                               var_prefill=var(self.prefill_new_tokens),
                               var_decode=var(self.decode_kv_tokens),
                               wall_time_s=wall_time_s)

    def summary(self) -> Dict[str, int]:
        return {
            "num_prefill": len(self.prefill_new_tokens),
            "prefill_new_tokens": sum(self.prefill_new_tokens),
            "prefill_cached_tokens": sum(self.prefill_cached_tokens),
            "num_decode": len(self.decode_kv_tokens),
            "decode_kv_tokens": sum(self.decode_kv_tokens),
        }


def phase_of(num_prefill: int, num_decode: int) -> str:
    if num_prefill and num_decode:
        return "mixed"
    return "prefill" if num_prefill else ("decode" if num_decode else "empty")


def fpm_from_counts(num_prefill: int, prefill_new_tokens: int,
                    prefill_cached_tokens: int, num_decode: int,
                    decode_kv_tokens: int, var_prefill: float = 0.0,
                    var_decode: float = 0.0, wall_time_s: float = 0.0) -> dict:
    """aiconfigurator / Dynamo ``ForwardPassMetrics`` from per-step totals.

    The same column names appear in ``--simulator-step-log`` CSVs, so a row
    of such a file can be priced with ``fpm_from_counts(**row_counts)``.
    """
    empty = {
        "num_prefill_requests": 0, "sum_prefill_tokens": 0,
        "var_prefill_length": 0.0, "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0, "var_decode_kv_tokens": 0.0,
    }
    return {
        "version": 1, "worker_id": "unicache", "dp_rank": 0,
        "counter_id": 0, "wall_time": float(wall_time_s),
        "scheduled_requests": {
            "num_prefill_requests": int(num_prefill),
            "sum_prefill_tokens": int(prefill_new_tokens),
            "var_prefill_length": float(var_prefill),
            "sum_prefill_kv_tokens": int(prefill_cached_tokens),
            "num_decode_requests": int(num_decode),
            "sum_decode_kv_tokens": int(decode_kv_tokens),
            "var_decode_kv_tokens": float(var_decode),
        },
        "queued_requests": empty,
    }


@dataclass
class Calibration:
    """Linear correction on top of aiconfigurator, per phase.

    Fitted by ``scripts/aic_timing/fit_calibration.py``; every number is
    readable: e.g. ``scale.decode = 1.3`` means real decode steps took 30%
    longer than the tables predict, ``overhead_ms.decode = 2.1`` is the fixed
    per-step CPU/scheduler cost the kernel tables do not include.
    """
    scale: Dict[str, float] = field(default_factory=lambda: {
        "prefill": 1.0, "decode": 1.0, "mixed": 1.0})
    overhead_ms: Dict[str, float] = field(default_factory=lambda: {
        "prefill": 0.0, "decode": 0.0, "mixed": 0.0})
    # Optional: charge this fraction of the simulator's own measured CPU time
    # per engine step (scheduler, block manager and eviction-policy calls) on
    # the virtual clock. On the GPU that CPU work is on the critical path; the
    # kernel tables omit it. Absent from a calibration JSON -> 0 (no change).
    cpu_scale: Dict[str, float] = field(default_factory=lambda: {
        "prefill": 0.0, "decode": 0.0, "mixed": 0.0})
    # Optional deterministic policy cost (sandboxed agent policies): milliseconds per IPC call
    # to the policy worker, and a factor on the worker's own process CPU time.
    policy_call_ms: float = 0.0
    policy_cpu_scale: float = 0.0
    # Optional engine-side costs the kernel tables omit (fitted from the synthetic sweep):
    # ms per cached block evicted during the step (block-manager CPU), and ms per request
    # still waiting in the scheduler queue during the step, per phase (scheduler CPU).
    evict_ms: float = 0.0
    waiting_ms: Dict[str, float] = field(default_factory=lambda: {
        "prefill": 0.0, "decode": 0.0, "mixed": 0.0})
    # Optional direct step model per phase, replacing scale * table + overhead:
    #   prefill: const + per_prompt * prompts + per_1k_new * new/1000 + per_1k_cached * cached/1000
    #            + per_1m_new_sq * sum(new_i^2)/1e6   (attention grows with prompt length squared)
    #   decode:  const + per_seq * sequences + per_1k_kv * context_tokens/1000
    linear: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Optional per-request arrival cost (prompt tokenization and input processing, done in the
    # engine's thread on a real server): const + per_1k * n/1000 + per_1m_sq * (n/1000)^2 ms,
    # n = prompt tokens. Charged on the virtual clock when the request is released.
    request_ms: Dict[str, float] = field(default_factory=dict)
    source: str = "uncalibrated"

    @classmethod
    def load(cls, path: Optional[str]) -> "Calibration":
        cal = cls()
        if not path:
            return cal
        with open(path) as f:
            data = json.load(f)
        cal.scale.update(data.get("scale", {}))
        cal.overhead_ms.update(data.get("overhead_ms", {}))
        cal.cpu_scale.update(data.get("cpu_scale", {}))
        cal.policy_call_ms = float(data.get("policy_call_ms", 0.0))
        cal.policy_cpu_scale = float(data.get("policy_cpu_scale", 0.0))
        cal.evict_ms = float(data.get("evict_ms", 0.0))
        cal.waiting_ms.update(data.get("waiting_ms", {}))
        cal.linear = dict(data.get("linear", {}))
        cal.request_ms = dict(data.get("request_ms", {}))
        cal.source = path
        return cal

    def cpu_charge(self, phase: str, cpu_ms: float) -> float:
        return self.cpu_scale.get(phase, 0.0) * cpu_ms

    def policy_charge(self, calls: int, worker_cpu_ms: float) -> float:
        return self.policy_call_ms * calls + self.policy_cpu_scale * worker_cpu_ms

    def engine_charge(self, phase: str, evictions: int, waiting: int) -> float:
        return self.evict_ms * evictions + self.waiting_ms.get(phase, 0.0) * waiting

    def request_charge(self, prompt_tokens: int) -> float:
        if not self.request_ms:
            return 0.0
        k = prompt_tokens / 1000.0
        return (self.request_ms.get("const", 0.0) + self.request_ms.get("per_1k", 0.0) * k
                + self.request_ms.get("per_1m_sq", 0.0) * k * k)

    def linear_ms(self, batch: "StepBatch") -> Optional[float]:
        m = self.linear.get(batch.phase)
        if not m:
            return None
        c = batch.summary()
        return (m.get("const", 0.0) + m.get("per_prompt", 0.0) * c["num_prefill"]
                + m.get("per_1k_new", 0.0) * c["prefill_new_tokens"] / 1000.0
                + m.get("per_1k_cached", 0.0) * c["prefill_cached_tokens"] / 1000.0
                + m.get("per_1m_new_sq", 0.0) * sum(n * n for n in batch.prefill_new_tokens) / 1e6
                + m.get("per_seq", 0.0) * c["num_decode"]
                + m.get("per_1k_kv", 0.0) * c["decode_kv_tokens"] / 1000.0)

    def apply(self, phase: str, raw_ms: float) -> float:
        return self.scale.get(phase, 1.0) * raw_ms + self.overhead_ms.get(
            phase, 0.0)


def literal_backend_version(system: str, backend: str) -> str:
    """Newest aiconfigurator data version shipped for (system, backend)."""
    from aiconfigurator_core.sdk import engine as aic_engine
    version = aic_engine._literal_backend_version(system, backend, None, None,
                                                  None)
    if not version:
        raise ValueError(f"aiconfigurator has no {backend} data for {system}")
    return version


def aic_engine_config(model_path: str, system: str, backend: str = "vllm",
                      backend_version: Optional[str] = None) -> dict:
    """Engine identity for a dense bf16 model on one GPU (tp=pp=1).

    Field names follow aiconfigurator's EngineConfig schema v1.
    ``model_path`` is a Hugging Face id or a local directory with config.json.
    """
    return {
        "schema_version": 1,
        "model_name": model_path,
        "system_name": system,
        "systems_path": None,
        "backend": backend,
        "backend_version": backend_version or literal_backend_version(
            system, backend),
        "kv_block_size": None,
        "tp_size": 1, "pp_size": 1, "attention_dp_size": 1,
        "moe_tp_size": 1, "moe_ep_size": 1, "cp_size": 1,
        "weight_dtype": "bfloat16", "moe_dtype": "bfloat16",
        "activation_dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
        "enable_shared_layer": True, "strict_provenance": False,
        "database_mode": "SILICON", "transfer_policy": None,
        "tolerate_dirless_version": False, "extra": {},
    }


class AicStepTiming:
    """Step latency from aiconfigurator's forward-pass model + calibration."""

    def __init__(self, model_path: str, system: str, backend: str = "vllm",
                 backend_version: Optional[str] = None,
                 calibration: Optional[Calibration] = None):
        from aiconfigurator_core.sdk.rust_engine_step import (
            RustForwardPassPerfModel)
        if os.path.isdir(model_path):
            model_path = os.path.abspath(model_path)
        self.config = aic_engine_config(model_path, system, backend,
                                        backend_version)
        self.model = RustForwardPassPerfModel.from_native(self.config)
        self.calibration = calibration or Calibration()
        self._cache: Dict[tuple, float] = {}
        # Fail now, not mid-run, if the tables cannot price this model.
        try:
            self.raw_ms(StepBatch(prefill_new_tokens=[128]))
            self.raw_ms(StepBatch(decode_kv_tokens=[128]))
        except Exception as exc:
            raise RuntimeError(
                f"aiconfigurator cannot price {model_path} on {system}: {exc}"
                "\nKnown gap: the a100_sxm vLLM tables have no attention data "
                "for head size 64 (Llama-3.2-1B). Use h100_sxm tables plus an "
                "A100 calibration file, or a model with head size 128 "
                "(e.g. Llama-3.1-8B). See docs/aic-timing.md.") from exc

    def describe(self) -> str:
        c = self.config
        return (f"aiconfigurator {c['backend']} {c['backend_version']} on "
                f"{c['system_name']} for {c['model_name']} "
                f"(calibration: {self.calibration.source})")

    def raw_ms(self, batch: StepBatch) -> float:
        """Uncalibrated aiconfigurator estimate in milliseconds."""
        return self.raw_ms_fpm(batch.to_fpm())

    def raw_ms_fpm(self, fpm: dict) -> float:
        req = fpm["scheduled_requests"]
        key = tuple(sorted(req.items()))
        if key not in self._cache:
            ms = self.model.estimate_forward_pass_time_ms(fpm)
            if ms is None:
                raise RuntimeError(
                    f"aiconfigurator returned no estimate for {req}")
            self._cache[key] = float(ms)
        return self._cache[key]

    def step_ms(self, batch: StepBatch) -> float:
        if batch.phase == "empty":
            return 0.0
        direct = self.calibration.linear_ms(batch)
        if direct is not None:
            return direct
        return self.calibration.apply(batch.phase, self.raw_ms(batch))
