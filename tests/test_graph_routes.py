import unittest
from device_route_status import DeviceRouteStatus


class GraphRoutingTelemetry(unittest.TestCase):
    def test_observed_per_pool_lifetime(self):
        status = DeviceRouteStatus()
        def create(phase):
            status.line(f"Inkling GPU-owned expert pools initialized: phase={phase} pools=2 charged_bytes=296048")
        status.line("Inkling GPU expert graph enabled: phase=1")
        create(1)
        self.assertFalse(status.snapshot()["device_routes_graph_active"])
        status.line("Inkling GPU expert graph dispatch observed")
        status.line("Inkling GPU expert routing executed without host lookup: phase=1")
        self.assertTrue(status.snapshot()["device_routes_graph_active"])
        status.line("Inkling GPU expert routing retired: phase=1")
        create(2)
        status.line("Inkling GPU expert graph unavailable: phase=2; using scheduler routing")
        status.line("Inkling GPU expert routing executed without host lookup: phase=2")
        state = status.snapshot()
        self.assertTrue(state["device_routes_active"])
        self.assertFalse(state["device_routes_graph_active"])
        self.assertTrue(state["device_routes_graph_observed"])
