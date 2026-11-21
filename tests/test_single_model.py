from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inkling import single_model_worker


class SingleModelTests(unittest.TestCase):
    def test_native_child_keeps_host_reservation_until_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with single_model_worker(root) as descriptor:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import sys; sys.stdin.read()"],
                    stdin=subprocess.PIPE, pass_fds=(descriptor,),
                )
            try:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with single_model_worker(root):
                        self.fail("A second controller acquired the native worker's reservation")
            finally:
                child.communicate(timeout=5)
            with single_model_worker(root) as descriptor:
                self.assertGreaterEqual(descriptor, 0)
