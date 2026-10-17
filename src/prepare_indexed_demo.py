"""Stage only an externally frozen, complete-package indexed-book demonstration."""

import argparse
import json
from pathlib import Path
import subprocess
import io
import zipfile

from src.indexed_demo_transport import (
    BOOKS,
    BRIDGE,
    TRANSPORT,
    at,
    authorize,
    digest,
    index_metadata,
    read,
    sha,
)


def prepare(
    repository, metadata, freeze_path, freeze_sha, index, sources, output, dataset
):
    repository, output = Path(repository).resolve(), Path(output)
    actual = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    if Path.cwd().resolve() != repository or actual != repository:
        raise ValueError("Stage from actual source repository root")
    if output.exists() or output.is_symlink():
        raise ValueError("Preserve existing demo stages")
    frozen = authorize(metadata, freeze_path, freeze_sha)
    files = {}
    for name in sorted(BRIDGE | TRANSPORT):
        raw = at(repository, name).read_bytes()
        if (
            sha(raw) != frozen["source_sha256"][name]
            or subprocess.check_output(["git", "show", "HEAD:" + name]) != raw
        ):
            raise ValueError("Actual committed expanded source differs")
        if name in TRANSPORT and b"\r" in raw:
            raise ValueError("Transport source requires LF bytes")
        files[name] = raw
    files["root-freeze.json"] = Path(freeze_path).read_bytes()
    for name in (
        "inference-contract.json",
        "joint-selection.json",
        "joint-status.json",
    ):
        files["package-metadata/" + name] = at(metadata, name).read_bytes()
    assets = {"assets/index.lrag": Path(index)} | {
        "assets/" + b + ".txt": at(sources, b + ".txt") for b in BOOKS
    }
    for name, path in assets.items():
        raw = path.read_bytes()
        if sha(raw) != frozen["assets"][name]:
            raise ValueError("Externally frozen existing demo asset differs")
        files[name] = raw
    index_metadata(index, {b: assets["assets/" + b + ".txt"] for b in BOOKS})
    # Recheck opaque input snapshots and authorization before writing any stage.
    authorize(metadata, freeze_path, freeze_sha)
    for name, path in assets.items():
        if digest(path) != sha(files[name]):
            raise ValueError("Demo asset changed during staging")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
    bundle_raw = bundle.getvalue()
    manifest = {
        "schema": "indexed_demo_manifest_v1",
        "archive_sha256": sha(bundle_raw),
        "root_freeze_sha256": freeze_sha,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "package_parent": frozen["package_parent"],
        "files": {name: sha(raw) for name, raw in files.items()},
    }
    manifest_raw = (json.dumps(manifest, indent=2) + "\n").encode()
    template = files["src/cloud/indexed_demo_runner.py"].decode()
    marker = "EXPECTED_MANIFEST_SHA256 = None"
    import ast

    tree = ast.parse(template)
    assignments = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "EXPECTED_MANIFEST_SHA256"
            for t in n.targets
        )
    ]
    if (
        len(assignments) != 1
        or not isinstance(assignments[0].value, ast.Constant)
        or assignments[0].value.value is not None
    ):
        raise ValueError("Unambiguous runner assignment required")
    rendered = template.replace(
        marker, "EXPECTED_MANIFEST_SHA256 = " + repr(sha(manifest_raw)), 1
    )
    output.mkdir(parents=True)
    upload = output / "upload"
    upload.mkdir()
    (upload / "indexed-demo-payload.zip").write_bytes(bundle_raw)
    (upload / "indexed-demo-manifest.json").write_bytes(manifest_raw)
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Private indexed literature demo source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )
    (output / "run.py").write_text(rendered, encoding="utf-8", newline="\n")
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "aikyatansinha/portfolio-indexed-literature-demo-v1",
                "title": "Indexed literature trained reader demo",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "kernel_sources": [frozen["package_parent"]],
                "competition_sources": [],
            },
            indent=2,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "metadata",
        "freeze",
        "freeze-sha256",
        "index",
        "sources",
        "output",
        "dataset",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(
        args.repository,
        args.metadata,
        args.freeze,
        args.freeze_sha256,
        args.index,
        args.sources,
        args.output,
        args.dataset,
    )


if __name__ == "__main__":
    main()
