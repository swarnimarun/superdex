from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from device_route_status import DeviceRouteStatus
from expert_cache_status import ExpertCacheStatus
from inkling import configure_device_routes


def allocation(size, slots):
    return (f"impl: CUDA0 MoE cache size = {size} MiB for 15066 MiB of host experts\n"
            f"impl: 20 physical layers, nvfp4: {slots} slots (10%)\n"
            "impl: 4 physical layers, q8_0: 128 slots (10%)\n")


def created(phase):
    return f"Inkling GPU-owned expert pools initialized: phase={phase} pools=2 charged_bytes=296048"


def executed(phase):
    return f"Inkling GPU expert routing executed without host lookup: phase={phase}"


def counts(phase, pool, visits, all_hits, hits, misses, byte_count):
    return (f"Inkling GPU expert counters: phase={phase} pool={pool} visits={visits} "
            f"all_hit={all_hits} hits={hits} misses={misses} bytes={byte_count}")


class DeviceRouteTests(unittest.TestCase):
    def test_native_owner_retirement(self):
        root = Path(__file__).resolve().parents[1]
        cc = shlex.split(os.environ.get("CXX", "c++"))
        if not shutil.which(cc[0]):
            self.skipTest("Requires a C++17 compiler")
        # Compile the exact methods installed by setup, using a CPU fake stream.
        patch = (root / "inkling-device-routes.patch").read_text()
        additions = "\n".join(line[1:] for line in patch.splitlines()
                              if line.startswith("+") and not line.startswith("+++"))
        start = additions.index("    void report_device_pool(")
        end = additions.index('stderr, "Inkling GPU expert routing retired:', start)
        end = additions.index("\n    }", end) + len("\n    }")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "retire.inc").write_text(additions[start:end])
            exe = work / "retire"
            subprocess.run([*cc, "-std=c++17", "-I", str(work), str(root / "tests/device_route_retire.cpp"),
                            "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True, capture_output=True)

    def test_optional_plugin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(configure_device_routes({}, path), (False, "off"))
            env = {"INKLING_DEVICE_ROUTES": "1", "LD_PRELOAD": "existing"}
            self.assertEqual(configure_device_routes(env, path), (True, "unavailable"))
            self.assertEqual(env, {"INKLING_DEVICE_ROUTES": "0", "LD_PRELOAD": "existing"})
            plugin = path / "libinkling-device-routes.so"
            plugin.touch()
            env["INKLING_DEVICE_ROUTES"] = "1"
            self.assertEqual(configure_device_routes(env, path), (True, "pending"))
            self.assertEqual(env["LD_PRELOAD"], f"{plugin}:existing")
            with self.assertRaises(RuntimeError):
                configure_device_routes({"INKLING_DEVICE_ROUTES": "yes"}, path)

    def test_charge_then_ownership_then_retirement(self):
        startup = allocation(1034.18, 175) + "initialized worker\n" + allocation(3078.32, 476)
        decode = ("Inkling phase cache: decode active, prefill scratch 3682.7 MiB, decode scratch 19.3 MiB, "
                  "expert cache 1034.2 -> 3078.6 MiB, shared ceiling 4716.9 MiB\n")
        phases = ExpertCacheStatus(startup + created(1) + "\n" + decode)
        stats = phases.snapshot()
        self.assertEqual(stats["expert_cache_slots"], 604)
        self.assertEqual(stats["device_routes_active"], False)
        self.assertEqual(stats["device_routes_metadata_bytes"], 296048)
        self.assertAlmostEqual(stats["expert_cache_allocated_mib"], 3078.32 + 296048 / 1048576)
        for line in [executed(1), counts(1,0,1,1,8,0,0), counts(1,1,2,0,12,4,4096)]:
            phases.phases.line(line)
        self.assertTrue(phases.snapshot()["device_routes_active"])
        self.assertEqual(phases.snapshot()["device_routes_session"]["visits"], 3)
        phases.phases.line("Inkling GPU expert routing retired: phase=1")
        self.assertEqual(phases.snapshot()["device_routes_metadata_bytes"], 0)
        self.assertFalse(phases.snapshot()["device_routes_active"])

    def test_sessions_include_final_snapshots_once(self):
        status = DeviceRouteStatus()
        for line in [created(1), executed(1), counts(1,0,128,20,800,224,448),
                     counts(1,0,256,41,1600,448,896), counts(1,0,128,20,800,224,448),
                     counts(1,1,2,0,8,8,16), "Inkling GPU expert routing retired: phase=1",
                     executed(1), created(1), counts(1,1,2,0,8,8,16), created(2), executed(2),
                     counts(2,1,4,1,16,16,32), counts(1,0,256,41,1600,448,896)]:
            status.line(line)
        self.assertTrue(status.snapshot()["device_routes_active"])
        self.assertEqual(status.snapshot()["device_routes_session"],
                         dict(visits=262, all_hit_visits=42, hits=1624, misses=472, miss_bytes=944))
        status.line("Inkling GPU expert routing retired: phase=2")
        status.line("Inkling GPU expert routing unavailable for this phase; using host lookup")
        self.assertEqual(status.snapshot()["device_routes_state"], "host")


if __name__ == "__main__":
    unittest.main()
