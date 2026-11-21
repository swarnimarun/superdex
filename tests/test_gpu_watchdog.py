import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from inkling import watch_gpu


class GpuWatchdogTests(unittest.TestCase):
    def read(self, reporter, observe=None, timeout=2):
        stop = threading.Event()
        readings, failures, processes = [], [], []
        popen = subprocess.Popen

        def spawn(command, **kwargs):
            self.assertEqual(command[-1], "--loop-ms=150")
            process = popen(command, **kwargs)
            processes.append(process)
            return process

        def record(free, used):
            readings.append((free, used))
            if observe:
                observe(stop, free, used)

        with patch("inkling.gpu_query", return_value=[sys.executable, "-c", reporter]) as query:
            with patch("inkling.subprocess.Popen", side_effect=spawn):
                watch_gpu("GPU-test", stop, record, failures.append, timeout)
        query.assert_called_once_with("GPU-test")
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        return readings, failures

    def test_continuous_samples_and_clean_stop(self):
        # Ordinary line-buffered stdout, including one line assembled in parts.
        reporter = ("import sys, time\n"
                    "print('GPU-test, 8000, 2000')\n"
                    "sys.stdout.write('GPU-test, '); sys.stdout.flush()\n"
                    "time.sleep(0.03); print('7900, 2100')\n"
                    "time.sleep(30)\n")
        def finish(stop, free, used):
            if used == 2100:
                stop.set()
        samples, errors = self.read(reporter, finish)
        self.assertEqual(samples, [(8000, 2000), (7900, 2100)])
        self.assertEqual(errors, [])

    def test_partial_samples_cannot_keep_an_unhealthy_reporter_alive(self):
        reporter = ("import os, time\nwhile True:\n"
                    "    os.write(1, b'0'); time.sleep(.02)\n")
        samples, errors = self.read(reporter, timeout=.2)
        self.assertFalse(samples)
        self.assertIn("No complete memory reading", errors[0])

    def test_read_errors_fail_closed_and_reap_reporter(self):
        for reporter, message in [
                ("pass", "memory stream ended"),
                ("print('GPU-other, 1, 2')", "while monitoring GPU-test"),
                ("print('NVML unavailable')", "NVML unavailable"),
                ("print('GPU-test, N/A, 3')", "No usable memory reading")]:
            with self.subTest(reporter=reporter):
                readings, failures = self.read(reporter)
                self.assertEqual(readings, [])
                self.assertEqual(len(failures), 1)
                self.assertIn(message, failures[0])

    def test_first_excessive_sample_triggers_failure(self):
        def budget(stop, free, used):
            if used > 2000:
                raise RuntimeError("device limit exceeded")
        samples, failures = self.read("print('GPU-test, 8000, 2000\\nGPU-test, 7999, 2001\\nGPU-test, 8000, 2000')", budget)
        self.assertEqual(samples, [(8000, 2000), (7999, 2001)])
        self.assertEqual(failures, ["device limit exceeded"])
