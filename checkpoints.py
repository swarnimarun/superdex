"""Verified mainline GGUF variants from the pinned Unsloth release."""
import argparse
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

RUNTIME = Path(__file__).resolve().parent / ".runtime"
REVISION = "ea0c79b2ecd73669e774c8519b79bfabfddbc027"


class Checkpoint(NamedTuple):
    path: Path
    quant: str
    digest: str


MODELS = {
    "reference": Checkpoint(RUNTIME / "inkling-reference.gguf", "UD-Q8_K_XL", "2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68"),
    "q4xl": Checkpoint(RUNTIME / "inkling-q4xl.gguf", "UD-Q4_K_XL", "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3"),
    "q4": Checkpoint(RUNTIME / "inkling.gguf", "UD-Q4_K_M", "78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae"),
    "iq3": Checkpoint(RUNTIME / "inkling-iq3.gguf", "UD-IQ3_XXS", "fd9ccedae17c1bb89744612fbb46730b5275f2a342133d0419776bd1ae311657"),
}

def verify_checkpoint(path, expected):
    """Hash a large GGUF without keeping previously read ranges in Linux's file cache."""
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        offset = 0
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass  # Cache eviction is advisory and can be unsupported on network filesystems.
            offset += len(block)
    return digest.hexdigest() == expected


def fetch(model):
    RUNTIME.mkdir(exist_ok=True)
    with model.path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if model.path.exists():
            if not verify_checkpoint(model.path, model.digest):
                raise SystemExit(f"Existing checkpoint failed its checksum: {model.path}")
            return
        partial = model.path.with_suffix(".gguf.part")
        url = (f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{REVISION}/"
               f"Mainline_Support/Inkling-2-Mini-{model.quant}.gguf_file")
        for _ in range(2):
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                            "--output", str(partial), url], check=True)
            if verify_checkpoint(partial, model.digest):
                partial.rename(model.path)
                return
            partial.unlink()
        raise SystemExit(f"Downloaded checkpoint failed its checksum: {model.quant}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=MODELS)
    fetch(MODELS[parser.parse_args().variant])
