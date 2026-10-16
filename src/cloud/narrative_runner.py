"""CPU runner importing plain Python from a private mounted Kaggle dataset.

The builder changes only configuration constants, never embeds encoded payloads.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

EXPECTED_MANIFEST_SHA256 = None
SMOKE = True


def mounted_source():
    manifests = list(Path("/kaggle/input").rglob("payload-provenance.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one private payload manifest")
    manifest_path = manifests[0]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Mounted payload manifest changed")
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["file_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid payload relative path")
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Mounted source/data file changed: " + name)
    if (source / "private-data/test").exists():
        raise ValueError("Training payload must not include test QA")
    return source, manifest


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.57.6", "sentencepiece==0.2.1"], check=True)
    source, manifest = mounted_source()
    sys.path.insert(0, str(source))
    from src.narrative_training import run
    output = Path("/kaggle/working/experiment")
    output.mkdir()
    (output / "payload-provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    exit_code = 1
    try:
        run(SimpleNamespace(data=source / "private-data", output=output / "run",
                            protocol=source / "docs/v5-training-protocol.json",
                            micro_batch=2, threads=4, smoke=SMOKE))
        exit_code = 0
    finally:
        (output / "runtime.json").write_text(json.dumps({
            "exit_code": exit_code, "wall_seconds": time.monotonic() - started,
            "cpu_count": os.cpu_count(), "threads": 4, "smoke": SMOKE,
            "test_files_packaged": False, "source_is_plain_mounted_python": True,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
