"""Follow cache allocation and phase-change messages from a single llama.cpp worker."""

from collections import Counter
from device_route_status import DeviceRouteStatus
import os
import re
import threading


_ALLOCATION = re.compile(r"CUDA0 MoE cache size =\s*([\d.]+) MiB for ([\d.]+) MiB of host experts")
_POOL = re.compile(r"(\d+) physical layers, (\w+):\s*(\d+) slots")
_DECODE = re.compile(r"Inkling phase cache: decode active, prefill scratch ([\d.]+) MiB, decode scratch ([\d.]+) MiB, "
                     r"expert cache [\d.]+ -> ([\d.]+) MiB")


class CachePhases:
    def __init__(self, startup):
        self.prefill = {}
        self.active = {}
        self.candidate = None
        self.prefill_scratch = None
        self.compact_prefill_kernel = None
        self.compact_decode_seen = False
        self.grouped_decode_seen = False
        self.elastic = {}
        self.controller_seen = False
        self.shortconv_seen = False
        self.routes = DeviceRouteStatus()
        for line in startup.splitlines():
            self.line(line)
        if not self.active and self.candidate:
            self._initial_allocation()

    def _initial_allocation(self):
        self.prefill = {**self.candidate, "expert_cache_phase": "prefill"}
        if not self.prefill["expert_cache_pools"]:
            self.prefill.pop("expert_cache_pools")
            self.prefill.pop("expert_cache_slots")
        self.active = self.prefill
        self.candidate = None

    def line(self, line):
        charged = self.routes.line(line)
        if charged > 0 and self.candidate:
            self.candidate["expert_cache_allocated_mib"] += charged / 1048576
        elif charged < 0 and self.active.get("expert_cache_phase") == "decode":
            self.active["expert_cache_allocated_mib"] += charged / 1048576
        if "Inkling elastic KV:" in line:
            if "using conventional KV" in line:
                self.elastic = {"elastic_global_kv_active": False}
            init = re.search(r"reserved=(\d+) MiB, (\d+) FP16 tensors, page=(\d+) KiB", line)
            if init:
                self.elastic = {"elastic_global_kv_active": True, "global_kv_reserved_mib": int(init[1]),
                                "global_kv_committed_mib": 0, "global_kv_page_kib": int(init[3])}
                chunk = re.search(r"chunk=(\d+) KiB", line)
                if chunk:
                    self.elastic["global_kv_chunk_kib"] = int(chunk[1])
            change = re.search(r"rows=(\d+) committed=(\d+) MiB reserved=(\d+) MiB", line)
            if change:
                self.elastic.update(elastic_global_kv_active=True, global_kv_target_rows=int(change[1]),
                                    global_kv_committed_mib=int(change[2]), global_kv_reserved_mib=int(change[3]))
        if "Inkling fused single-token short convolution executed" in line:
            self.shortconv_seen = True
        if "Inkling four-kernel QKV loop controllers active" in line:
            self.controller_seen = True
        if "Inkling compact global attention executed:" in line:
            match = re.search(r"\bkernel=(native|hand)\b", line)
            if match:
                self.compact_prefill_kernel = match[1]
        if "Inkling compact grouped-query decode executed:" in line:
            self.grouped_decode_seen = True
        if "Inkling compact single-token vector attention executed:" in line:
            self.compact_decode_seen = True
        if "MoE cache size =" in line:
            match = _ALLOCATION.search(line)
            if match:
                if self.candidate and not self.active:
                    self._initial_allocation()
                self.candidate = {"expert_cache_allocated_mib": float(match[1]),
                                  "host_expert_weights_mib": float(match[2]),
                                  "expert_cache_slots": 0, "expert_cache_pools": []}
            return
        if self.candidate is not None and "physical layers," in line:
            match = _POOL.search(line)
            if match:
                slots = int(match[3])
                self.candidate["expert_cache_slots"] += slots
                self.candidate["expert_cache_pools"].append(
                    {"layers": int(match[1]), "weight_type": match[2], "slots": slots})
            return
        if "Inkling phase cache:" not in line:
            if not self.active and self.candidate:
                self._initial_allocation()
            return
        if "restored prefill" in line:
            if not self.prefill and self.candidate:
                self._initial_allocation()
            self.active = {**self.prefill, "expert_cache_phase": "prefill"}
            if self.prefill_scratch is not None:
                self.active["expert_cache_compute_scratch_mib"] = self.prefill_scratch
            self.candidate = None
            return
        match = _DECODE.search(line)
        if match:
            self.prefill_scratch = float(match[1])
            size = float(match[3])
            banks = self.candidate or {}
            # The marker rounds to one decimal and the bank message to two.
            # If the file was truncated, the marker still proves the active size.
            if abs(banks.get("expert_cache_allocated_mib", -1) - size) > 0.061:
                banks = {"expert_cache_allocated_mib": size}
            expected = Counter((p["layers"], p["weight_type"]) for p in self.prefill.get("expert_cache_pools", ()))
            observed = Counter((p["layers"], p["weight_type"]) for p in banks.get("expert_cache_pools", ()))
            if expected - observed:
                banks = {key: val for key, val in banks.items() if key not in ("expert_cache_pools", "expert_cache_slots")}
            self.active = {**banks, "expert_cache_phase": "decode", "expert_cache_compute_scratch_mib": float(match[2])}
            self.candidate = None


class ExpertCacheStatus:
    """Read a growing log incrementally and publish the last completed phase transition."""

    chunk = 64 * 1024

    def __init__(self, startup, path=None, offset=0, dynamic=None):
        self.phases = CachePhases(startup)
        self.path = path
        self.dynamic = path is not None if dynamic is None else dynamic
        self.offset = offset
        self.partial = b""
        self.skip_long_line = False
        self.lock = threading.Lock()

    def update(self):
        if self.path is None:
            return False
        with self.lock, self.path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < self.offset:
                self.offset, self.partial, self.skip_long_line = 0, b"", False
                self.phases.candidate = None
            if size == self.offset:
                return False
            stream.seek(self.offset)
            data = stream.read(self.chunk)
            self.offset += len(data)
            lines = (self.partial + data).split(b"\n")
            self.partial = lines.pop()
            if self.skip_long_line and lines:
                lines.pop(0)
                self.skip_long_line = False
            if len(self.partial) > 4096:
                self.partial = b""
                self.skip_long_line = True
            for line in lines:
                self.phases.line(line.decode(errors="replace"))
            return self.offset < size

    def follow(self, stop):
        while not stop.is_set():
            if not self.update():
                stop.wait(0.25)

    def snapshot(self):
        behind = self.update()
        with self.lock:
            result = dict(self.phases.active)
            if not self.dynamic and result.get("expert_cache_phase") == "prefill":
                result["expert_cache_phase"] = "fixed"
            if self.phases.compact_prefill_kernel:
                result["compact_prefill_kernel_observed"] = self.phases.compact_prefill_kernel
            if self.phases.shortconv_seen:
                result["shortconv_observed"] = "sm120-four-tap"
            if self.phases.controller_seen:
                result["controller_fusion_observed"] = "sm120-qkv"
            if self.phases.grouped_decode_seen:
                result["grouped_decode_kernel_observed"] = "vector-gqa3"
            if self.phases.compact_decode_seen:
                result["compact_decode_kernel_observed"] = "vector"
            result.update(self.phases.elastic)
            result.update(self.phases.routes.snapshot())
            if behind:
                result["expert_cache_stats_catching_up"] = True
            return result


def expert_cache_stats(log):
    return CachePhases(log).active
