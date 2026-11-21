import tempfile
import unittest
from pathlib import Path

from inkling import configure_compact_attention, configure_controller, configure_shortconv


class CompactLauncherTests(unittest.TestCase):
    def test_optional_native_plugin_and_explicit_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = {"INKLING_COMPACT_KERNEL": "auto"}
            self.assertEqual(configure_compact_attention(env, directory), "dense")
            self.assertEqual(env["INKLING_COMPACT_FA"], "0")
            with self.assertRaisesRegex(RuntimeError, "build-compact-fa --native"):
                configure_compact_attention({"INKLING_COMPACT_KERNEL": "native"}, directory)

            dispatch = directory / "libinkling-compact-dispatch.so"
            native = directory / "libinkling-native-fa.so"
            dispatch.touch()
            native.touch()
            env = {"INKLING_COMPACT_KERNEL": "auto", "LD_PRELOAD": "profiler.so"}
            self.assertEqual(configure_compact_attention(env, directory), "native")
            self.assertEqual(env["INKLING_NATIVE_FA_LIBRARY"], str(native))
            self.assertEqual(env["LD_PRELOAD"], f"{dispatch}:profiler.so")
            self.assertEqual(env["INKLING_COMPACT_FA"], "1")

    def test_controller_is_optional_and_preserves_other_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = {"LD_PRELOAD": "profiler.so"}
            self.assertEqual(configure_controller(env, directory), "off")
            self.assertEqual(env["LD_PRELOAD"], "profiler.so")
            env["INKLING_CONTROLLER_FUSION"] = "1"
            self.assertEqual(configure_controller(env, directory), "unavailable")
            self.assertEqual(env["INKLING_CONTROLLER_FUSION"], "0")
            self.assertEqual(env["LD_PRELOAD"], "profiler.so")
            plugin = directory / "libinkling-controller.so"
            plugin.touch()
            env["INKLING_CONTROLLER_FUSION"] = "1"
            self.assertEqual(configure_controller(env, directory), "sm120-qkv")
            self.assertEqual(env["LD_PRELOAD"], f"{plugin}:profiler.so")
            with self.assertRaisesRegex(RuntimeError, "must be 0 or 1"):
                configure_controller({"INKLING_CONTROLLER_FUSION": "yes"}, directory)

    def test_shortconv_preload_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = {"LD_PRELOAD": "profiler.so"}
            self.assertEqual(configure_shortconv(env, directory), "off")
            env["INKLING_FUSED_SHORTCONV"] = "1"
            self.assertEqual(configure_shortconv(env, directory), "unavailable")
            self.assertEqual(env["LD_PRELOAD"], "profiler.so")
            self.assertEqual(env["INKLING_FUSED_SHORTCONV"], "0")
            plugin = directory / "libinkling-shortconv.so"
            plugin.touch()
            env["INKLING_FUSED_SHORTCONV"] = "1"
            self.assertEqual(configure_shortconv(env, directory), "sm120-four-tap")
            self.assertEqual(env["LD_PRELOAD"], f"{plugin}:profiler.so")

    def test_hand_kernel_and_disable_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = {"INKLING_COMPACT_KERNEL": "hand", "INKLING_NATIVE_FA_LIBRARY": "old-native.so"}
            files = [directory / n for n in ("libinkling-compact-dispatch.so", "libinkling-compact-fa.so")]
            for file in files:
                file.touch()
            self.assertEqual(configure_compact_attention(env, directory), "hand")
            self.assertEqual(env["LD_PRELOAD"], ":".join(map(str, files)))
            self.assertNotIn("INKLING_NATIVE_FA_LIBRARY", env)
            disabled = {"INKLING_COMPACT_KERNEL": "native", "INKLING_COMPACT_FA": "0"}
            self.assertEqual(configure_compact_attention(disabled, directory), "dense")
            self.assertNotIn("LD_PRELOAD", disabled)
