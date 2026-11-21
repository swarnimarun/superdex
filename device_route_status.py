"""Accumulate device expert-routing counters from one native worker's log."""

from collections import Counter
import re


class DeviceRouteStatus:
    # The native worker emits monotonic counters per (phase, pool). Deferred
    # samples and final snapshots use the same format. Log replay is idempotent.
    allocation = re.compile(r"GPU-owned expert pools initialized: phase=(\d+) pools=(\d+) charged_bytes=(\d+)")
    activation = re.compile(r"GPU expert routing executed without host lookup: phase=(\d+)")
    retirement = re.compile(r"GPU expert routing retired: phase=(\d+)")
    counters = re.compile(r"GPU expert counters: phase=(\d+) pool=(\d+) visits=(\d+) "
                          r"all_hit=(\d+) hits=(\d+) misses=(\d+) bytes=(\d+)")
    keys = ("visits", "all_hit_visits", "hits", "misses", "miss_bytes")

    def __init__(self):
        self.phase = 0
        self.graph_phase = None
        self.graph_observed = False
        self.state = None
        self.latest = {}
        self.retired = False
        self.charged = 0
        self.total = Counter()

    def line(self, line):
        """Return the change in allocated device metadata bytes."""
        if "Inkling GPU" not in line:
            return 0
        allocated = self.allocation.search(line)
        if allocated:
            phase, _, charged = map(int, allocated.groups())
            if phase > self.phase:
                self.phase = phase
                self.state = "initialized"
                self.latest.clear()
                self.retired = False
                self.charged = charged
                return charged
            return 0
        if "GPU expert graph enabled: phase=" in line:
            phase = re.search(r"phase=(\d+)", line)
            if phase:
                reported = int(phase[1])
                if reported > self.phase or (reported == self.phase and not self.retired):
                    self.graph_phase = reported
            return 0
        if "GPU expert graph dispatch observed" in line:
            self.graph_observed = True
            return 0
        if "GPU expert routing unavailable for this phase" in line:
            self.state = "host"
            self.charged = 0
            return 0
        sample = self.counters.search(line)
        if sample:
            phase, pool, *values = map(int, sample.groups())
            if phase == self.phase and not self.retired:
                previous = self.latest.get(pool, [0] * len(values))
                if values[0] >= previous[0]:
                    self.total.update(dict(zip(self.keys, (v - p for v, p in zip(values, previous)))))
                    self.latest[pool] = values
        elif (active := self.activation.search(line)) and int(active[1]) == self.phase and not self.retired:
            self.state = "active"
        elif (done := self.retirement.search(line)) and int(done[1]) == self.phase:
            self.retired = True
            self.state = "host"
            released = self.charged
            self.charged = 0
            return -released
        return 0

    def snapshot(self):
        if self.state is None:
            return {}
        return {"device_routes_graph_observed": self.graph_observed,
                "device_routes_graph_active": self.graph_phase == self.phase and self.state == "active",
                "device_routes_state": self.state, "device_routes_active": self.state == "active",
                "device_routes_metadata_bytes": self.charged,
                "device_routes_session": {key: self.total[key] for key in self.keys}}
