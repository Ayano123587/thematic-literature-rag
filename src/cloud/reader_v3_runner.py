"""Readable runner for retrospective v3 reproduction on mounted private files.

Historical v3 outcomes remain anchored to their original source/payload hashes.
This transport change does not rerun or alter those outcomes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

EXPECTED_MANIFEST_SHA256 = None


def main():
    os.environ["READER_STARTED_MONOTONIC"] = str(time.monotonic())
    watchdog = threading.Timer(900, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.57.6", "sentence-transformers==5.1.2"], check=True)
    manifests = list(Path("/kaggle/input").rglob("reader-v3-provenance.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one private v3 reproduction manifest")
    path = manifests[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Private manifest changed")
    source = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["file_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid source relative path")
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Mounted v3 file changed: " + name)
    sys.path.insert(0, str(source))
    import reader_cloud
    reader_cloud.run(source, Path("/kaggle/working"), 900)
    watchdog.cancel()


if __name__ == "__main__":
    main()
