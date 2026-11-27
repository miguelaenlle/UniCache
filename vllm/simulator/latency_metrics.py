# SPDX-License-Identifier: Apache-2.0
"""Per-request latency from the simulator clock.

Works the same in virtual time (simulation) and wall time (GPU replay), so the
two can be compared request by request.

Definitions (seconds):
    ttft  first output token time - arrival time (includes queueing)
    itl   gap between consecutive output tokens of one request (pooled)
    tpot  mean itl of one request
    e2e   finish time - arrival time

Arrival is the trace time at which the simulator released the request (for a
follow-up turn: previous turn's completion plus its think time).
"""
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class _Req:
    arrival: float
    prompt_tokens: int = 0
    first_token: Optional[float] = None
    last_token: Optional[float] = None
    finish: Optional[float] = None
    n_tokens: int = 0
    gaps: List[float] = field(default_factory=list)


class LatencyRecorder:

    def __init__(self):
        self.reqs: Dict[str, _Req] = {}

    def on_arrival(self, request_id: str, arrival_time: float):
        self.reqs[request_id] = _Req(arrival=float(arrival_time))

    def on_step(self, request_outputs, now: float):
        """Call after the clock has advanced past one engine step."""
        for out in request_outputs:
            req = self.reqs.get(getattr(out, "request_id", None))
            if req is None or not getattr(out, "outputs", None):
                continue
            n = len(out.outputs[0].token_ids)
            if n > req.n_tokens:
                if req.first_token is None:
                    req.first_token = now
                    req.prompt_tokens = len(out.prompt_token_ids or [])
                else:
                    # One gap per step; a step emits one token per sequence.
                    req.gaps.append(now - req.last_token)
                req.last_token = now
                req.n_tokens = n
            if out.finished and req.finish is None:
                req.finish = now

    def rows(self) -> List[dict]:
        rows = []
        for rid, r in self.reqs.items():
            if r.finish is None or r.first_token is None:
                continue
            rows.append({
                "request_id": rid,
                "arrival_s": r.arrival,
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.n_tokens,
                "ttft_s": r.first_token - r.arrival,
                "tpot_s": float(np.mean(r.gaps)) if r.gaps else 0.0,
                "e2e_s": r.finish - r.arrival,
            })
        return rows

    def summary(self) -> dict:
        rows = self.rows()
        gaps = [g for r in self.reqs.values() if r.finish is not None
                for g in r.gaps]

        def stats(xs):
            if not len(xs):
                return None
            a = np.asarray(xs, dtype=float)
            return {"mean": float(a.mean()),
                    "p50": float(np.percentile(a, 50)),
                    "p90": float(np.percentile(a, 90)),
                    "p99": float(np.percentile(a, 99))}

        out = {
            "completed_requests": len(rows),
            "unfinished_requests": len(self.reqs) - len(rows),
            "ttft_s": stats([r["ttft_s"] for r in rows]),
            "itl_s": stats(gaps),
            "tpot_s": stats([r["tpot_s"] for r in rows if r["output_tokens"] > 1]),
            "e2e_s": stats([r["e2e_s"] for r in rows]),
        }
        if rows:
            start = min(r["arrival_s"] for r in rows)
            end = max(r["arrival_s"] + r["e2e_s"] for r in rows)
            out["makespan_s"] = end - start
            out["output_tokens_per_s"] = (sum(r["output_tokens"] for r in rows)
                                          / max(end - start, 1e-9))
        return out

    def write(self, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        rows = self.rows()
        with open(os.path.join(out_dir, "latency_requests.csv"), "w",
                  newline="") as f:
            fields = ["request_id", "arrival_s", "prompt_tokens",
                      "output_tokens", "ttft_s", "tpot_s", "e2e_s"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        summary = self.summary()
        with open(os.path.join(out_dir, "latency_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary


class StepLog:
    """One CSV row per engine step: batch shape, modelled and measured time.

    In GPU replay ``measured_ms`` is the wall time of ``engine.step()``; these
    rows are the training data for ``scripts/aic_timing/fit_calibration.py``.
    """
    FIELDS = ["step", "t_s", "phase", "num_prefill", "prefill_new_tokens",
              "prefill_cached_tokens", "num_decode", "decode_kv_tokens",
              "model_ms", "measured_ms", "cpu_ms", "policy_calls", "policy_events",
              "policy_cpu_ms", "evictions", "num_waiting"]

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._n = 0

    def write(self, t_s: float, batch, model_ms, measured_ms, cpu_ms=None, policy=None,
              engine=None):
        if batch.phase == "empty":
            return
        # cpu_ms: wall time of engine.step() in this process. In simulation it
        # is CPU-side work only (scheduler, block manager, policy calls); in
        # GPU replay it equals measured_ms.
        row = {"step": self._n, "t_s": t_s, "phase": batch.phase,
               "model_ms": model_ms, "measured_ms": measured_ms, "cpu_ms": cpu_ms}
        if policy is not None:
            row["policy_calls"], row["policy_events"], row["policy_cpu_ms"] = policy
        if engine is not None:
            row["evictions"], row["num_waiting"] = engine
        row.update(batch.summary())
        self._w.writerow(row)
        self._n += 1

    def close(self):
        self._f.close()
