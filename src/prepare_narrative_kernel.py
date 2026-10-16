"""Prepare readable Python and separately mounted private narrative data."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from src.data.prepare_fairytale import sha


def prepare(data, output, smoke, dataset):
    data, output = Path(data), Path(output)
    if not dataset or "/" not in dataset:
        raise ValueError("A private Kaggle dataset slug is required")
    if output.exists():
        raise ValueError("Refusing to overwrite kernel payload")
    paths = ["src/__init__.py", "src/data/__init__.py", "src/data/prepare_fairytale.py",
             "src/narrative_training.py", "src/models/__init__.py",
             "src/models/narrative_generator.py", "evaluation_scripts/predict_narrative.py",
             "evaluation_scripts/evaluate_narrative.py", "train.py",
             "docs/v5-training-protocol.json"]
    files = {name: Path(name).read_bytes() for name in paths}
    for path in data.rglob("*"):
        if path.is_file() and "test" not in path.relative_to(data).parts:
            files["private-data/" + path.relative_to(data).as_posix()] = path.read_bytes()
    manifest = json.loads(files["private-data/manifest.json"])
    provenance = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "executed_protocol_sha256": sha(files["docs/v5-training-protocol.json"]),
        "staging_protocol_sha256": manifest["protocol_sha256"],
        "file_sha256": {name: sha(raw) for name, raw in files.items()},
        "smoke": smoke, "test_files_packaged": False,
        "hardware": "Free CPU only; GPU quota exhausted; enable_gpu=false",
        "transport": "Plain mounted source files; standard ZIP upload automatically extracted by Kaggle",
    }
    files["payload-provenance.json"] = json.dumps(provenance, indent=2).encode()
    output.mkdir(parents=True)
    upload = output / "dataset-upload"
    upload.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for name, raw in files.items():
            handle.writestr(name, raw)
    license_path = next(Path("data/external/fairytaleqa").glob("FairytaleQAData-*/LICENSE"))
    (upload / "LICENSE").write_bytes(license_path.read_bytes())
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset, "title": "Private narrative training source-family study",
        "licenses": [{"name": "other"}],
        "description": "Private FairytaleQA Apache-2.0 subset and plain training Python. See LICENSE. No test QA included.",
    }, indent=2), encoding="utf-8")
    runner = Path("src/cloud/narrative_runner.py").read_text(encoding="utf-8")
    runner = runner.replace("EXPECTED_MANIFEST_SHA256 = None",
                            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(files["payload-provenance.json"])))
    runner = runner.replace("SMOKE = True", "SMOKE = " + repr(smoke))
    (output / "run.py").write_text(runner, encoding="utf-8", newline="\n")
    slug = "portfolio-narrative-t5-cpu-smoke" if smoke else "portfolio-narrative-t5-cpu-train"
    metadata = {
        "id": "aikyatansinha/" + slug, "title": slug.replace("-", " ").title(),
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {**provenance, "private_dataset": dataset,
              "payload_zip_sha256": sha((upload / "payload.zip").read_bytes()),
              "private_bootstrap_sha256": sha((output / "run.py").read_bytes()),
              "bootstrap_bytes": (output / "run.py").stat().st_size}
    kind = "smoke" if smoke else "train"
    Path(f"docs/v5-{kind}-preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"private_kernel": metadata["id"], "runner_bytes": report["bootstrap_bytes"],
                      "test_files_packaged": False, "gpu": False, "plain_source": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    prepare(args.data, args.output, args.smoke, args.dataset)
