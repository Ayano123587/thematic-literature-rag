"""Plain-source CPU runner for real narrative evidence reranker adaptation."""
import json
import hashlib
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
    path = manifests[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Mounted payload manifest changed")
    source = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
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
                    "transformers==4.57.6"], check=True)
    # Verify source bytes before importing anything from the mounted dataset.
    source, manifest = mounted_source()
    sys.path.insert(0, str(source))
    from src.reranker_training import run
    output = Path("/kaggle/working/experiment")
    output.mkdir()
    (output / "payload-provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    began = time.monotonic()
    exit_code = 1
    try:
        run(SimpleNamespace(data=source / "private-data", output=output / "run",
                            protocol=source / "docs/v5-reranker-protocol.json", smoke=SMOKE))
        exit_code = 0
    finally:
        (output / "runtime.json").write_text(json.dumps({
            "exit_code": exit_code, "wall_seconds": time.monotonic() - began,
            "cpu_count": os.cpu_count(), "threads": 4, "smoke": SMOKE,
            "test_files_packaged": False, "source_is_plain_mounted_python": True,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
