"""Build a private dataset of plain source files and a readable CPU runner."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from src.data.prepare_fairytale import sha


def prepare(data, output, dataset, smoke, protocol="docs/v5-reranker-protocol.json"):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a prepared reranker payload")
    paths = ["src/__init__.py", "src/data/__init__.py", "src/data/prepare_fairytale.py",
             "src/narrative_training.py", "src/reranker_training.py", "src/cloud/__init__.py",
             "src/cloud/narrative_runner.py", "docs/v5-reranker-protocol.json"]
    files = {name: Path(name).read_bytes() for name in paths}
    # Keep the runner's mounted protocol filename stable while selecting an
    # explicit preregistered experiment; legacy default payloads are unchanged.
    files["docs/v5-reranker-protocol.json"] = Path(protocol).read_bytes()
    for path in data.rglob("*"):
        if path.is_file() and "test" not in path.relative_to(data).parts:
            files["private-data/" + path.relative_to(data).as_posix()] = path.read_bytes()
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "protocol_sha256": sha(files["docs/v5-reranker-protocol.json"]),
              "file_sha256": {name: sha(raw) for name, raw in files.items()},
              "test_files_packaged": False, "smoke": smoke, "gpu": False,
              "protocol_source_path": str(protocol),
              "transport": "Plain mounted Python, JSON and CSV; ZIP extracted by Kaggle"}
    files["payload-provenance.json"] = json.dumps(report, indent=2).encode()
    output.mkdir(parents=True)
    upload = output / "dataset-upload"
    upload.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    license_path = next(Path("data/external/fairytaleqa").glob("FairytaleQAData-*/LICENSE"))
    (upload / "LICENSE").write_bytes(license_path.read_bytes())
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset, "title": "Private narrative section reranker study",
        "licenses": [{"name": "other"}],
        "description": "Private FairytaleQA Apache-2.0 training/validation subset and readable source. No test QA; see LICENSE.",
    }, indent=2), encoding="utf-8")
    runner = Path("src/cloud/reranker_runner.py").read_text(encoding="utf-8")
    runner = runner.replace("EXPECTED_MANIFEST_SHA256 = None",
                            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(files["payload-provenance.json"])))
    runner = runner.replace("SMOKE = True", "SMOKE = " + repr(smoke))
    (output / "run.py").write_text(runner, encoding="utf-8", newline="\n")
    slug = "portfolio-narrative-reranker-cpu-" + ("smoke" if smoke else "train")
    metadata = {"id": "aikyatansinha/" + slug, "title": slug.replace("-", " ").title(),
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_internet": True,
                "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report.update({"private_dataset": dataset, "runner_sha256": sha((output / "run.py").read_bytes()),
                   "payload_zip_sha256": sha((upload / "payload.zip").read_bytes())})
    kind = "smoke" if smoke else "train"
    preflight = Path(f"docs/v5-reranker-{kind}-preflight.json") if protocol == "docs/v5-reranker-protocol.json" else Path(
        "docs") / (Path(protocol).stem + f"-{kind}-preflight.json")
    if preflight.exists() and protocol != "docs/v5-reranker-protocol.json":
        raise ValueError("Preserve prior experiment preflight; use a versioned protocol")
    preflight.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": metadata["id"], "test_packaged": False, "gpu": False,
                      "runner_bytes": (output / "run.py").stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", default="docs/v5-reranker-protocol.json")
    args = parser.parse_args()
    prepare(args.data, args.output, args.dataset, args.smoke, args.protocol)
