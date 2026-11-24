"""Tests for step timing, latency metrics and calibration fitting.

    .venv-sim312/bin/python -m unittest scripts.aic_timing.test_aic_timing
"""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vllm.simulator.latency_metrics import LatencyRecorder, StepLog  # noqa: E402
from vllm.simulator.step_timing import (Calibration, StepBatch,  # noqa: E402
                                        fpm_from_counts)

HAS_AIC = importlib.util.find_spec("aiconfigurator_core") is not None
MODEL_1B = str(ROOT / "models/Llama-3.2-1B-Instruct")
MODEL_8B = str(ROOT / "models/Llama-3.1-8B-Instruct")


def meta(is_prompt, chunk, computed_blocks=0, lengths=(10, )):
    seq_data = {i: SimpleNamespace(get_len=lambda n=n: n)
                for i, n in enumerate(lengths)}
    return SimpleNamespace(is_prompt=is_prompt, token_chunk_size=chunk,
                           computed_block_nums=list(range(computed_blocks)),
                           seq_data=seq_data)


def output(rid, n_tokens, finished=False, prompt_len=5):
    return SimpleNamespace(request_id=rid, finished=finished,
                           prompt_token_ids=[0] * prompt_len,
                           outputs=[SimpleNamespace(token_ids=[0] * n_tokens)])


class StepBatchTest(unittest.TestCase):

    def test_prefill_counts_cached_blocks_as_hits(self):
        b = StepBatch.from_metadata([meta(True, 100, computed_blocks=3)], 8)
        self.assertEqual((b.prefill_new_tokens, b.prefill_cached_tokens),
                         ([76], [24]))
        self.assertEqual(b.phase, "prefill")

    def test_fully_cached_prompt_still_computes_one_token(self):
        b = StepBatch.from_metadata([meta(True, 16, computed_blocks=2)], 8)
        self.assertEqual((b.prefill_new_tokens, b.prefill_cached_tokens),
                         ([1], [15]))

    def test_decode_reads_full_context(self):
        b = StepBatch.from_metadata(
            [meta(False, 1, lengths=(40, )), meta(False, 1, lengths=(60, ))], 8)
        self.assertEqual(b.decode_kv_tokens, [40, 60])
        self.assertEqual(b.phase, "decode")
        self.assertEqual(StepBatch.from_metadata([], 8).phase, "empty")

    def test_fpm_matches_counts(self):
        b = StepBatch([10, 30], [8, 0], [])
        req = b.to_fpm()["scheduled_requests"]
        self.assertEqual(req["sum_prefill_tokens"], 40)
        self.assertEqual(req["sum_prefill_kv_tokens"], 8)
        self.assertEqual(req["var_prefill_length"], 100.0)
        self.assertEqual(fpm_from_counts(2, 40, 8, 0, 0)["scheduled_requests"]
                         ["num_prefill_requests"], 2)


class CalibrationTest(unittest.TestCase):

    def test_default_is_identity_and_file_overrides(self):
        self.assertEqual(Calibration().apply("decode", 5.0), 5.0)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"scale": {"decode": 2.0},
                       "overhead_ms": {"decode": 1.5}}, f)
            f.flush()
            cal = Calibration.load(f.name)
        self.assertEqual(cal.apply("decode", 5.0), 11.5)
        self.assertEqual(cal.apply("prefill", 5.0), 5.0)


class LatencyRecorderTest(unittest.TestCase):

    def test_ttft_itl_e2e(self):
        rec = LatencyRecorder()
        rec.on_arrival("a", 1.0)
        rec.on_step([output("a", 1)], 1.5)        # first token
        rec.on_step([output("a", 2)], 1.6)
        rec.on_step([output("a", 3, finished=True)], 1.8)
        row, = rec.rows()
        self.assertAlmostEqual(row["ttft_s"], 0.5)
        self.assertAlmostEqual(row["tpot_s"], 0.15)
        self.assertAlmostEqual(row["e2e_s"], 0.8)
        self.assertEqual(row["output_tokens"], 3)
        s = rec.summary()
        self.assertEqual(s["completed_requests"], 1)
        self.assertAlmostEqual(s["itl_s"]["mean"], 0.15)

    def test_unfinished_requests_are_excluded(self):
        rec = LatencyRecorder()
        rec.on_arrival("a", 0.0)
        rec.on_step([output("a", 1)], 0.2)
        self.assertEqual(rec.rows(), [])
        self.assertEqual(rec.summary()["unfinished_requests"], 1)

    def test_step_log_skips_empty_steps(self):
        with tempfile.TemporaryDirectory() as d:
            log = StepLog(f"{d}/steps.csv")
            log.write(0.0, StepBatch(), None, None)
            log.write(0.1, StepBatch([5], [3], []), 1.25, 2.0)
            log.close()
            with open(f"{d}/steps.csv") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prefill_cached_tokens"], "3")


@unittest.skipUnless(HAS_AIC, "aiconfigurator not installed")
class AicTimingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from vllm.simulator.step_timing import AicStepTiming
        cls.AicStepTiming = AicStepTiming
        cls.h100_1b = AicStepTiming(MODEL_1B, "h100_sxm")

    def test_cache_hits_make_prefill_cheaper(self):
        t = self.h100_1b
        miss = t.step_ms(StepBatch([2048], [0], []))
        hit = t.step_ms(StepBatch([128], [1920], []))
        self.assertLess(hit, miss / 2)

    def test_bigger_batches_cost_more(self):
        t = self.h100_1b
        self.assertLess(t.step_ms(StepBatch([], [], [1024])),
                        t.step_ms(StepBatch([], [], [1024] * 64)))
        self.assertLess(t.step_ms(StepBatch([256], [0], [])),
                        t.step_ms(StepBatch([4096], [0], [])))

    def test_gpu_choice_matters(self):
        a100 = self.AicStepTiming(MODEL_8B, "a100_sxm")
        h100 = self.AicStepTiming(MODEL_8B, "h100_sxm")
        batch = StepBatch([], [], [2048] * 16)
        self.assertGreater(a100.step_ms(batch), h100.step_ms(batch))

    def test_calibration_is_applied(self):
        cal = Calibration(scale={"decode": 2.0}, overhead_ms={"decode": 1.0})
        t = self.AicStepTiming(MODEL_1B, "h100_sxm", calibration=cal)
        batch = StepBatch([], [], [512] * 4)
        self.assertAlmostEqual(t.step_ms(batch),
                               2.0 * self.h100_1b.raw_ms(batch) + 1.0)

    def test_fit_calibration_recovers_known_parameters(self):
        """Synthetic 'GPU' = 1.5 x tables + 2 ms (decode), 1.2 x + 0.5 (prefill)."""
        t = self.h100_1b
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "steps.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=StepLog.FIELDS)
                w.writeheader()
                step = 0
                for n in [32, 128, 512, 2048, 4096]:
                    for cached in [0, 512]:
                        b = StepBatch([n], [cached], [])
                        w.writerow({"step": step, "phase": "prefill",
                                    "measured_ms": 1.2 * t.raw_ms(b) + 0.5,
                                    **b.summary()})
                        step += 1
                for bs in [1, 4, 16, 64]:
                    for ctx in [256, 2048]:
                        b = StepBatch([], [], [ctx] * bs)
                        w.writerow({"step": step, "phase": "decode",
                                    "measured_ms": 1.5 * t.raw_ms(b) + 2.0,
                                    **b.summary()})
                        step += 1
            out = Path(d) / "cal.json"
            subprocess.run([sys.executable,
                            str(ROOT / "scripts/aic_timing/fit_calibration.py"),
                            "--steps", str(path), "--model", MODEL_1B,
                            "--system", "h100_sxm", "--out", str(out)],
                           check=True, capture_output=True)
            cal = json.loads(out.read_text())
        self.assertAlmostEqual(cal["scale"]["decode"], 1.5, places=3)
        self.assertAlmostEqual(cal["overhead_ms"]["decode"], 2.0, places=3)
        self.assertAlmostEqual(cal["scale"]["prefill"], 1.2, places=3)
        self.assertAlmostEqual(cal["overhead_ms"]["prefill"], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
