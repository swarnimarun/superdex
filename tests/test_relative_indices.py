import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


class RelativeIndexTests(unittest.TestCase):
    def test_portable_indices_and_mask(self):
        root = Path(__file__).resolve().parents[1]
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not shutil.which(compiler[0]) or not shutil.which("patch"):
            self.skipTest("Requires a C++17 compiler and patch")
        # Compile the same header setup installs. This runs without a model or CUDA.
        patch = (root / "inkling-relative-indices.patch").read_text()
        header = next(hunk for hunk in re.split(r"(?=^--- )", patch, flags=re.M)
                      if "+++ b/src/inkling-relative-indices.h\n" in hunk)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "src").mkdir()
            subprocess.run(["patch", "--quiet", "--batch", "--fuzz=0", "-p1"], cwd=work,
                           input=header, text=True, check=True)
            binary = work / "relative-indices"
            subprocess.run([*compiler, "-std=c++17", "-O2", "-I", str(work / "src"),
                            str(root / "tests/relative_indices.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
