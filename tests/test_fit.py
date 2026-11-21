import re
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inkling import HOST_EXPERT_WEIGHTS_RULE, HOST_OUTPUT_RULE, MODEL, effective_load_mode, expert_cache_stats, fit, fit_with_reserve, host_attention_devices


def tensor_overrides(command):
    return command[command.index("-ot") + 1].split(",") if "-ot" in command else []


def offloads_all_expert_weights(command):
    return HOST_EXPERT_WEIGHTS_RULE in tensor_overrides(command)


def estimate(command, **kwargs):
    gpu_layers = int(command[command.index("-ngl") + 1])
    if offloads_all_expert_weights(command):
        host_experts = 24
    elif "-ot" in command:
        layer_range = re.search(r"\(([\d|]+)\)", command[command.index("-ot") + 1])
        host_experts = len(layer_range[1].split("|")) if layer_range else 0
    else:
        host_experts = 0
    # Shared graph scratch, per-layer dense/KV memory, and static expert weights.
    total = 200 + 100 * gpu_layers + 30 * (24 - host_experts)
    return subprocess.CompletedProcess(command, 0, f"CUDA0 0 {total} 0\n", "")


class PlacementTests(unittest.TestCase):
    def test_expert_slot_counts_are_shared_across_physical_layers(self):
        summary = expert_cache_stats(
            "CUDA0 MoE cache size =  1784.06 MiB for 15066.00 MiB of host experts\n"
            "20 physical layers, nvfp4:   173 slots (6.8%, 1.1 tokens worth)\n"
            " 4 physical layers, q8_0:   128 slots (25.0%, 4.0 tokens worth)\n")
        self.assertEqual(summary["expert_cache_slots"], 301)
        self.assertEqual(summary["expert_cache_allocated_mib"], 1784.06)
        self.assertEqual(expert_cache_stats("no on-demand cache"), {})

    def setUp(self):
        self.args = SimpleNamespace(model=MODEL, context=131072, batch=64, ubatch=32, threads=8, batch_threads=None, load_mode="mmap",
                                    cache_type_k="f16", cache_type_v="f16", reserve_mib=500, mode="auto",
                                    cache_mib=None, host_global_kv=0, stage_global_kv="auto", host_output=0)

    def test_host_kv_staging_uses_the_selected_graph_for_fit_and_launch(self):
        self.args.host_global_kv = 4
        self.args.mode = "cache"
        def estimate_staging(command, **kwargs):
            staged = kwargs["env"]["INKLING_STAGE_GLOBAL_KV"] == "1"
            size = int(command[command.index("--moe-cache-mib") + 1]) if "--moe-cache-mib" in command else 0
            parts = (100, 100 + size, 600 if staged else 100)
            return subprocess.CompletedProcess(command, 0, "CUDA0 %d %d %d\n" % parts, "")
        with patch("inkling.subprocess.run", side_effect=estimate_staging):
            for mode, available, selected in [("auto", 1800, "1"), ("auto", 1200, "0"), ("0", 1800, "0")]:
                with self.subTest(mode=mode, available=available):
                    self.args.stage_global_kv = mode
                    env = {}
                    _, projected, reserve = fit_with_reserve(self.args, env, available)
                    self.assertEqual(env["INKLING_STAGE_GLOBAL_KV"], selected)
                    self.assertLessEqual(projected + reserve, available)
            self.args.stage_global_kv = "1"
            with self.assertRaisesRegex(RuntimeError, "exceed the available budget"):
                fit_with_reserve(self.args, {}, 1200)

    def test_reported_host_attention_follows_backend_assignment_and_flash_fallback(self):
        lines = ("Inkling global KV: layer=5 buffer=CUDA_Host attention=CUDA0\n"
                 "Inkling global KV: layer=5 buffer=CUDA_Host attention=CUDA0\n"
                 "Inkling global KV: layer=11 device=CPU\n")
        self.assertEqual(host_attention_devices(lines), {"CUDA0": 1, "CPU": 1})
        self.assertEqual(host_attention_devices(lines + "Flash Attention not supported, set to disabled\n"), {"CPU": 2})
        self.assertEqual(host_attention_devices(""), {})

    def test_direct_io_stats_include_open_and_read_fallbacks(self):
        direct = "impl: file 'model.gguf' opened with direct I/O (alignment: 4096 bytes)\n"
        read_fallback = "read_raw_unsafe: file 'model.gguf' falling back to buffered I/O due to Invalid argument\n"
        open_fallback = "Failed to open file 'shard.gguf' with unbuffered I/O: Invalid argument. Falling back to buffered I/O\n"
        self.assertEqual(effective_load_mode("dio", direct), "dio")
        self.assertEqual(effective_load_mode("dio", direct + read_fallback), "none")
        self.assertEqual(effective_load_mode("dio", open_fallback), "none")
        self.assertEqual(effective_load_mode("dio", direct + open_fallback), "mixed")
        self.assertEqual(effective_load_mode("mmap", ""), "mmap")

    def test_cache_can_fit_when_uncached_transfer_staging_cannot(self):
        self.args.mode = "auto"
        self.args.batch = self.args.ubatch = 512
        def cuda_estimate(command, **kwargs):
            if not offloads_all_expert_weights(command):
                parts = (19000, 4299, 800)
            elif "--moe-cache-mib" not in command:
                parts = (2202, 4299, 7300)
            else:
                size = int(command[command.index("--moe-cache-mib") + 1])
                parts = (2202, 4299 + size, 1400)
            self.assertEqual(kwargs["env"]["INKLING_MOE_CACHE_UBATCH_FLOOR"], command[command.index("-ub") + 1])
            return subprocess.CompletedProcess(command, 0, "CUDA0 %d %d %d\n" % parts, "")
        with patch("inkling.subprocess.run", side_effect=cuda_estimate):
            flags, projected = fit(self.args, {}, available=9500)
        self.assertEqual(flags[flags.index("--moe-cache-mib") + 1], "1099")
        self.assertEqual(projected, 9000)

    def test_adaptive_reserve_rejects_tiny_gpu_batches(self):
        self.args.batch = self.args.ubatch = 512
        self.args.host_global_kv = self.args.reserve_mib = None
        def cuda_estimate(command, **kwargs):
            host = int(kwargs["env"]["INKLING_HOST_GLOBAL_KV"])
            tile = int(command[command.index("-ub") + 1])
            size = int(command[command.index("--moe-cache-mib") + 1]) if "--moe-cache-mib" in command else 0
            if not offloads_all_expert_weights(command):
                parts = (18000, 4300, 1000)
            else:
                parts = (2200, 4300 - 512*host + size, 2250 if host == 0 and tile >= 64 else 400)
            return subprocess.CompletedProcess(command, 0, "CUDA0 %d %d %d\n" % parts, "")
        for available, expected_host, expected_reserve in [(9861, 0, 768), (9500, 2, 1280)]:
            env = {}
            with self.subTest(available=available), patch("inkling.subprocess.run", side_effect=cuda_estimate):
                flags, projected, reserve = fit_with_reserve(self.args, env, available)
                self.assertEqual(reserve, expected_reserve)
                self.assertEqual(env["INKLING_HOST_GLOBAL_KV"], str(expected_host))
                self.assertEqual(flags[flags.index("-ngl") + 1], "49")
                self.assertEqual(flags[flags.index("-ub") + 1], "512")
                self.assertLessEqual(projected, available - reserve)
                self.assertIsNone(self.args.reserve_mib)

    def test_auto_prefers_host_output_to_host_attention(self):
        self.args.batch = self.args.ubatch = 512
        self.args.host_output = self.args.host_global_kv = self.args.reserve_mib = None
        def cuda_estimate(command, **kwargs):
            host_kv = int(kwargs["env"]["INKLING_HOST_GLOBAL_KV"])
            tile = int(command[command.index("-ub") + 1])
            host_output = HOST_OUTPUT_RULE in tensor_overrides(command)
            cache = int(command[command.index("--moe-cache-mib") + 1]) if "--moe-cache-mib" in command else 0
            if not offloads_all_expert_weights(command):
                parts = (18000, 4300, 1000)
            else:
                parts = (1318 if host_output else 2202, 4300 - 512*host_kv + cache, 2250 if tile >= 64 else 400)
            return subprocess.CompletedProcess(command, 0, "CUDA0 %d %d %d\n" % parts, "")
        for available, host_setting, manual_reserve, host_attention, host_output, reserve in [
                (9700, None, None, 0, True, 1536), (9700, 0, None, 2, False, 1280),
                (9700, 1, None, 0, True, 1536), (9500, None, None, 2, False, 1280),
                (9500, 1, 768, 0, True, 768)]:
            with self.subTest(available=available, host_setting=host_setting), patch("inkling.subprocess.run", side_effect=cuda_estimate):
                self.args.host_output, self.args.reserve_mib = host_setting, manual_reserve
                env = {}
                flags, projected, effective = fit_with_reserve(self.args, env, available)
                self.assertEqual(HOST_OUTPUT_RULE in tensor_overrides(flags), host_output)
                self.assertEqual(env["INKLING_HOST_GLOBAL_KV"], str(host_attention))
                self.assertEqual(flags[flags.index("-ub") + 1], "512")
                self.assertEqual(effective, reserve)
                self.assertLessEqual(projected, available - reserve)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_scale_tensors_follow_layer_placement(self, _run):
        self.args.mode = "cache"
        flags, _ = fit(self.args, {}, available=6500)
        self.assertNotIn("-cmoe", flags)
        rule = flags[flags.index("-ot") + 1].split(",")[0]
        pattern, device = rule.rsplit("=", 1)
        self.assertEqual(device, "CPU")
        self.assertRegex("blk.17.ffn_gate_up_exps.weight", pattern)
        self.assertRegex("blk.8.ffn_down_exps.weight", pattern)
        self.assertNotRegex("blk.17.ffn_gate_up_exps.scale", pattern)
        self.assertNotRegex("blk.8.ffn_down_exps.input_scale", pattern)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_forced_host_output_combines_with_explicit_expert_overrides(self, _run):
        self.args.host_output = 1
        for mode in ("static", "hybrid"):
            self.args.mode = mode
            with self.subTest(mode=mode):
                flags, _ = fit(self.args, {}, available=5950 if mode == "static" else 6310)
                self.assertEqual(flags.count("-ot"), 1)
                rules = flags[flags.index("-ot") + 1].split(",")
                self.assertIn(HOST_OUTPUT_RULE, rules)
                self.assertTrue(any("ffn_" in rule for rule in rules))

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_reserve_respects_capacity_and_explicit_settings(self, _run):
        for available, explicit_reserve, host, expected in [
                (9399, None, None, 1280), (9400, None, None, 768),
                (9800, None, 4, 1280), (9800, 1200, None, 1200), (9200, 768, 2, 768)]:
            with self.subTest(available=available, explicit=explicit_reserve, host=host):
                self.args.reserve_mib, self.args.host_global_kv = explicit_reserve, host
                env = {}
                _, projected, reserve = fit_with_reserve(self.args, env, available)
                self.assertEqual(reserve, expected)
                self.assertEqual(env["INKLING_HOST_GLOBAL_KV"], str(host or 0))
                self.assertLessEqual(projected, available - reserve)

    def test_low_cap_tries_host_kv_before_tiny_batches(self):
        self.args.batch = self.args.ubatch = 512
        self.args.host_global_kv = None
        env = {"LLAMA_ARG_MOE_CACHE_CPU_SHARE": "1"}
        checked = []
        def estimate_low_cap(command, **kwargs):
            host = int(kwargs["env"]["INKLING_HOST_GLOBAL_KV"])
            tile = int(command[command.index("-ub") + 1])
            checked.append((host, tile, tuple(command)))
            if not offloads_all_expert_weights(command):
                parts = (2000, 100, 50)
            elif "--moe-cache-mib" not in command:
                parts = (100, 100, 1200)
            else:
                cache = int(command[command.index("--moe-cache-mib") + 1])
                parts = (100, 100 + cache, 900 if host == 0 and tile > 2 else 100)
            return subprocess.CompletedProcess(command, 0, "CUDA0 %d %d %d\n" % parts, "")
        with patch("inkling.subprocess.run", side_effect=estimate_low_cap):
            flags, projected = fit(self.args, env, available=1500)
        self.assertEqual(env["INKLING_HOST_GLOBAL_KV"], "2")
        self.assertEqual(flags[flags.index("-ub") + 1], "512")
        self.assertEqual(flags[flags.index("--moe-cache-cpu-share") + 1], "0")
        self.assertLessEqual(projected, 1000)
        self.assertEqual(len(checked), len(set(checked)))
        # The four unsuccessful large batches share one base memory measurement.
        base = [t for h, t, cmd in checked if h == 0 and offloads_all_expert_weights(cmd) and "--moe-cache-mib" not in cmd]
        self.assertEqual(base, [1])
        # An explicit host placement still accepts the tiny working batch.
        self.args.host_global_kv = 0
        with patch("inkling.subprocess.run", side_effect=estimate_low_cap):
            flags, projected = fit(self.args, env, available=1500)
        self.assertEqual(env["INKLING_HOST_GLOBAL_KV"], "0")
        self.assertEqual(flags[flags.index("-ub") + 1], "2")
        self.assertLessEqual(projected, 1000)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_splits_layers_then_fits_experts_under_the_same_reserve(self, _run):
        flags, projected = fit(self.args, {}, available=4440)
        self.assertEqual(flags[flags.index("-ngl") + 1], "37")
        self.assertIn("-ot", flags)
        self.assertEqual(projected, 3930)
        self.assertLessEqual(projected, 4440 - self.args.reserve_mib)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_full_model_stays_on_gpu_when_it_fits(self, run):
        self.args.batch, self.args.ubatch = 512, 128
        flags, _ = fit(self.args, {}, available=7000)
        for command in (flags, run.call_args.args[0]):
            self.assertEqual(command[command.index("-b") + 1], "512")
            self.assertEqual(command[command.index("-ub") + 1], "128")
        self.assertEqual(flags[flags.index("-ngl") + 1], "49")
        self.assertNotIn("-ot", flags)
        self.assertEqual(run.call_count, 1)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_insufficient_budget_still_fails(self, _run):
        with self.assertRaisesRegex(RuntimeError, "minimum buffers"):
            fit(self.args, {}, available=600)

    @patch("inkling.subprocess.run", side_effect=estimate)
    def test_explicit_static_mode_keeps_its_layer_placement(self, _run):
        self.args.mode = "static"
        with self.assertRaisesRegex(RuntimeError, "Dense weights and context"):
            fit(self.args, {}, available=4440)
