"""Package ignored narrative data into a private CPU-only Kaggle experiment."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import subprocess
import zipfile

from src.data.prepare_fairytale import sha


def prepare(data, output, smoke, dataset=None):
    data, output = Path(data), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite kernel payload")
    source = [Path("src/__init__.py"), Path("src/data/__init__.py"),
              Path("src/data/prepare_fairytale.py"), Path("src/narrative_training.py"),
              Path("train.py"), Path("docs/v5-training-protocol.json")]
    files = {str(p).replace("\\", "/"): p.read_bytes() for p in source}
    for path in data.rglob("*"):
        if path.is_file() and "test" not in path.relative_to(data).parts:
            files["private-data/" + path.relative_to(data).as_posix()] = path.read_bytes()
    manifest = json.loads(files["private-data/manifest.json"])
    protocol_hash = sha(files["docs/v5-training-protocol.json"])
    # Staging source text/QA is invariant across a prospective config amendment;
    # preserve both hashes rather than rewrite the source manifest retrospectively.
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "executed_protocol_sha256": protocol_hash,
              "staging_protocol_sha256": manifest["protocol_sha256"],
              "file_sha256": {name: sha(raw) for name, raw in files.items()},
              "smoke": smoke, "test_files_packaged": False,
              "hardware": "free CPU only; GPU quota exhausted; enable_gpu=false"}
    files["payload-provenance.json"] = json.dumps(report, indent=2).encode()
    packed = {name: base64.b64encode(gzip.compress(raw)).decode() for name, raw in files.items()}
    bootstrap = '''import os,sys,subprocess,json,base64,gzip,time
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','sentencepiece==0.2.1'],check=True)
root=Path('/kaggle/working/experiment');root.mkdir()
packed=PACKED
for name,raw in packed.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gzip.decompress(base64.b64decode(raw)))
os.chdir(root)
started=time.monotonic()
command=[sys.executable,'train.py','--data','private-data','--output','run','--micro-batch','2','--threads','4']+SMOKE
result=subprocess.run(command)
(root/'runtime.json').write_text(json.dumps({'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,'test_files_packaged':False}))
if result.returncode: raise SystemExit(result.returncode)
'''.replace("PACKED", repr(packed)).replace("SMOKE", repr(["--smoke"] if smoke else []))
    output.mkdir(parents=True)
    if dataset:
        dataset_folder = output / "dataset-upload"
        dataset_folder.mkdir()
        archive = dataset_folder / "payload.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for name, raw in files.items():
                handle.writestr(name, raw)
        license_file = next(Path("data/external/fairytaleqa").glob("FairytaleQAData-*/LICENSE"))
        (dataset_folder / "LICENSE").write_bytes(license_file.read_bytes())
        (dataset_folder / "dataset-metadata.json").write_text(json.dumps({
            "id": dataset, "title": "Private narrative training source-family study",
            "licenses": [{"name": "other"}],
            "description": "Private FairytaleQA Apache-2.0 research subset and original training source. See LICENSE. No test QA included."
        }, indent=2), encoding="utf-8")
        bootstrap = '''import os,sys,subprocess,json,time,zipfile,hashlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','sentencepiece==0.2.1'],check=True)
archives=list(Path('/kaggle/input').rglob('payload.zip'))
if len(archives)!=1: raise ValueError('Expected one private payload')
if hashlib.sha256(archives[0].read_bytes()).hexdigest()!=EXPECTED_HASH: raise ValueError('Payload bytes changed')
root=Path('/kaggle/working/experiment');root.mkdir()
with zipfile.ZipFile(archives[0]) as handle: handle.extractall(root)
os.chdir(root)
started=time.monotonic()
command=[sys.executable,'train.py','--data','private-data','--output','run','--micro-batch','2','--threads','4']+SMOKE_ARGUMENTS
result=subprocess.run(command)
(root/'runtime.json').write_text(json.dumps({'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,'test_files_packaged':False}))
if result.returncode: raise SystemExit(result.returncode)
'''.replace("EXPECTED_HASH", repr(sha(archive.read_bytes()))).replace(
            "SMOKE_ARGUMENTS", repr(["--smoke"] if smoke else []))
        report["private_dataset"] = dataset
        report["payload_zip_sha256"] = sha(archive.read_bytes())
    (output / "run.py").write_text(bootstrap, encoding="utf-8", newline="\n")
    slug = "portfolio-narrative-t5-cpu-smoke" if smoke else "portfolio-narrative-t5-cpu-train"
    metadata = {"id": "aikyatansinha/" + slug, "title": slug.replace("-", " ").title(),
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_internet": True,
                "dataset_sources": [dataset] if dataset else [], "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report["private_bootstrap_sha256"] = sha((output / "run.py").read_bytes())
    report["bootstrap_bytes"] = (output / "run.py").stat().st_size
    Path("docs/v5-" + ("smoke" if smoke else "train") + "-preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"private_kernel": metadata["id"], "bootstrap_bytes": report["bootstrap_bytes"],
                      "test_files_packaged": False, "gpu": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", help="Private Kaggle dataset slug for payloads exceeding script size limits")
    args = parser.parse_args()
    prepare(args.data, args.output, args.smoke, args.dataset)
