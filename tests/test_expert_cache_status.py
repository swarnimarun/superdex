from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from expert_cache_status import ExpertCacheStatus, expert_cache_stats
from inkling import configure_prompt_lookup, env_integer


def allocation(size, nv_slots, q_slots):
    return (f"impl: CUDA0 MoE cache size =  {size:.2f} MiB for 15066.00 MiB of host experts\n"
            f"impl: 20 physical layers, nvfp4: {nv_slots} slots (10%)\n"
            f"impl: 4 physical layers, q8_0: {q_slots} slots (10%)\n")


BASE = allocation(1034.18, 175, 34)
LARGE = allocation(3078.32, 476, 128)
DECODE = ("Inkling phase cache: decode active, prefill scratch 3682.7 MiB, decode scratch 19.3 MiB, "
          "expert cache 1034.2 -> 3078.3 MiB, shared ceiling 4716.9 MiB\n")
RESTORE = "Inkling phase cache: restored prefill expert cache request 1037.0 MiB\n"


class CacheStatusTests(unittest.TestCase):
    def test_virtual_and_physical_kv_accounting(self):
        status = ExpertCacheStatus("Inkling elastic KV: reserved=4096 MiB, 16 FP16 tensors, page=2048 KiB chunk=8192 KiB\n")
        self.assertEqual(status.snapshot()["global_kv_chunk_kib"], 8192)
        self.assertEqual(status.snapshot()["global_kv_committed_mib"], 0)
        status.phases.line("Inkling elastic KV: rows=20480 committed=640 MiB reserved=4096 MiB")
        self.assertEqual(status.snapshot()["global_kv_committed_mib"], 640)
        self.assertEqual(status.snapshot()["global_kv_reserved_mib"], 4096)
        status.phases.line("Inkling elastic KV: rows=0 committed=0 MiB reserved=4096 MiB")
        self.assertEqual(status.snapshot()["global_kv_committed_mib"], 0)
        status.phases.line("Inkling elastic KV: using conventional KV for this device, placement or layout")
        self.assertEqual(status.snapshot()["elastic_global_kv_active"], False)
        self.assertNotIn("global_kv_reserved_mib", status.snapshot())

    def test_decode_cache_limits(self):
        for name in ("INKLING_DECODE_CACHE_EXTRA_MIB", "INKLING_DECODE_CACHE_HEADROOM_MIB"):
            with self.subTest(name=name):
                self.assertEqual(env_integer({}, name, 0, 0, 16384), 0)
                for value in ("0", "1024", "16384"):
                    self.assertEqual(env_integer({name: value}, name, 0, 0, 16384), int(value))
                for value in ("-1", "16385", "1.5", ""):
                    with self.assertRaisesRegex(RuntimeError, name):
                        env_integer({name: value}, name, 0, 0, 16384)

    def test_one_model_lookup_flags_and_phase_threshold(self):
        env = {"LLAMA_ARG_SPEC_TYPE": "draft-simple", "LLAMA_ARG_SPEC_DRAFT_MODEL": "external.gguf",
               "LLAMA_ARG_SPEC_DRAFT_HF_REPO": "external/repo"}
        flags, stats = configure_prompt_lookup(env)
        self.assertEqual(flags, [])
        self.assertEqual(stats["prompt_lookup"], "none")
        self.assertEqual(env, {"INKLING_LOOKUP_ROLLBACK_ROWS": "0", "INKLING_DECODE_CACHE_MIN_TOKENS": "12"})
        for mode in ("ngram-simple", "ngram-map-k"):
            env = {"INKLING_PROMPT_LOOKUP": mode, "INKLING_DECODE_CACHE_MIN_TOKENS": "4"}
            flags, stats = configure_prompt_lookup(env)
            self.assertEqual(flags, ["--spec-type", mode, f"--spec-{mode}-size-n", "4", f"--spec-{mode}-size-m", "4"])
            self.assertEqual(env["INKLING_LOOKUP_ROLLBACK_ROWS"], "4")
            self.assertEqual(stats["expert_cache_decode_min_tokens"], 4)
        control = {"INKLING_LOOKUP_ROLLBACK_ROWS": "4"}
        self.assertEqual(configure_prompt_lookup(control)[0], [])
        self.assertEqual(control["INKLING_LOOKUP_ROLLBACK_ROWS"], "4")

    def test_invalid_lookup_settings(self):
        for name, value in (("INKLING_PROMPT_LOOKUP", "draft-simple"), ("INKLING_LOOKUP_ROLLBACK_ROWS", "0"),
                            ("INKLING_LOOKUP_ROLLBACK_ROWS", "8"), ("INKLING_LOOKUP_MATCH_TOKENS", "5"),
                            ("INKLING_DECODE_CACHE_MIN_TOKENS", "0"), ("INKLING_DECODE_CACHE_MIN_TOKENS", str(2**31))):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(RuntimeError, name):
                configure_prompt_lookup({"INKLING_PROMPT_LOOKUP": "ngram-simple", name: value})

    def test_latest_completed_transition_and_return_to_prefill(self):
        startup = BASE + "ready\n"
        decode = expert_cache_stats(startup + LARGE + DECODE)
        self.assertEqual(decode["expert_cache_allocated_mib"], 3078.32)
        self.assertEqual(decode["expert_cache_slots"], 604)
        self.assertEqual(decode["expert_cache_phase"], "decode")
        self.assertEqual(decode["expert_cache_compute_scratch_mib"], 19.3)
        self.assertEqual(expert_cache_stats(startup + LARGE)["expert_cache_slots"], 209)
        self.assertEqual(expert_cache_stats(BASE + LARGE + DECODE)["expert_cache_slots"], 604)
        self.assertEqual(expert_cache_stats(LARGE + DECODE)["expert_cache_slots"], 604)
        restored = expert_cache_stats(startup + LARGE + DECODE + BASE + RESTORE)
        self.assertEqual(restored["expert_cache_phase"], "prefill")
        self.assertEqual(restored["expert_cache_allocated_mib"], 1034.18)
        self.assertEqual(restored["expert_cache_compute_scratch_mib"], 3682.7)
        self.assertEqual([p["slots"] for p in restored["expert_cache_pools"]], [175, 34])

    def test_split_lines_truncation_and_missing_pool(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"worker.log"
            path.write_text(BASE)
            status = ExpertCacheStatus(BASE, path, path.stat().st_size)
            with path.open("a") as f:
                f.write(LARGE + DECODE[:22])
            self.assertEqual(status.snapshot()["expert_cache_phase"], "prefill")
            with path.open("a") as f:
                f.write(DECODE[22:])
            self.assertEqual(status.snapshot()["expert_cache_slots"], 604)
            path.write_text(RESTORE)
            self.assertEqual(status.snapshot()["expert_cache_slots"], 209)
            path.write_text("")
            self.assertEqual(status.snapshot()["expert_cache_phase"], "prefill")
            path.write_text(LARGE.splitlines(keepends=True)[0] + DECODE)
            measured = status.snapshot()
            self.assertEqual(measured["expert_cache_allocated_mib"], 3078.32)
            self.assertNotIn("expert_cache_slots", measured)

    def test_bounded_read_and_no_cache(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"worker.log"
            path.write_text(BASE + "request\n"*30000 + LARGE + DECODE)
            status = ExpertCacheStatus(BASE, path, len(BASE))
            view = status.snapshot()
            self.assertTrue(view["expert_cache_stats_catching_up"])
            self.assertEqual(status.offset, len(BASE)+status.chunk)
            while status.update():
                pass
            self.assertEqual(status.snapshot()["expert_cache_slots"], 604)
        self.assertEqual(expert_cache_stats("worker without a cache\n"), {})
        self.assertEqual(ExpertCacheStatus(BASE).snapshot()["expert_cache_phase"], "fixed")
