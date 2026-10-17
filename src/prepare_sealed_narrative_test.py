"""Prepare one private sealed release only after an actual root policy/package freeze."""

import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from src.sealed_narrative_test import (
    authorize_freeze,
    reserve,
    cohort_metadata,
    sha,
    digest,
    at,
    consume_dispatch,
)

LICENSE_SHA256 = "335d2c093cbb191de9693d87c9935f832a2442a67615a6e47ac71f74ccb4bd5d"


def prepare(
    source_root, metadata, root_freeze, root_sha256, data, ledger, output, dataset
):
    repository = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    if Path.cwd().resolve() != repository:
        raise ValueError("Stage from repository root")
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve sealed release stage")
    frozen = authorize_freeze(metadata, root_freeze, root_sha256)
    license_file = Path(metadata) / "LICENSE"
    if not license_file.is_file() or digest(license_file) != LICENSE_SHA256:
        raise ValueError("Actual upstream license required before release reservation")
    reservation = reserve(ledger, frozen, root_sha256)
    cohort = cohort_metadata(data, reservation)
    package_contract = json.loads(at(metadata, "inference-contract.json").read_bytes())
    files = {}
    for name, expected in package_contract["source_sha256"].items():
        raw = at(source_root, name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Actual frozen inference source differs")
        files["src/" + name] = raw
    for name in (
        "src/sealed_narrative_test.py",
        "src/cloud/sealed_narrative_test_runner.py",
        "src/prepare_sealed_narrative_test.py",
    ):
        raw = Path(name).read_bytes()
        if b"\r" in raw:
            raise ValueError("Sealed transport source must use LF bytes")
        if sha(raw) != frozen["release_source_sha256"][name]:
            raise ValueError("Externally frozen sealed transport source differs")
        if subprocess.check_output(["git", "show", "HEAD:" + name]) != raw:
            raise ValueError("Uncommitted sealed transport source")
        files[name] = raw
    files["root-freeze.json"] = Path(root_freeze).read_bytes()
    files["NOTICE/FairytaleQA-LICENSE"] = license_file.read_bytes()
    files["package-metadata/inference-contract.json"] = at(
        metadata, "inference-contract.json"
    ).read_bytes()
    files["package-metadata/joint-selection.json"] = at(
        metadata, "joint-selection.json"
    ).read_bytes()
    files["cohort.json"] = (json.dumps(cohort, indent=2) + "\n").encode()
    for name, expected in cohort["files"].items():
        raw = at(data, name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Sealed input changed during staging")
        files["private-test/" + name] = raw
    ticket = {
        "schema": "sealed_narrative_test_ticket_v1",
        "root_freeze_sha256": root_sha256,
        "package_contract_sha256": frozen["package_contract_sha256"],
        "joint_result_sha256": frozen["joint_result_sha256"],
        "selected_arm": frozen["selected_arm"],
        "package_parent": frozen["package_parent"],
        "cohort_sha256": sha(json.dumps(cohort, sort_keys=True).encode()),
        "reservation_sha256": digest(reservation),
        "source_sha256": {
            name: sha(raw) for name, raw in files.items() if name.startswith("src/")
        },
        "runtime": frozen["runtime"],
        "questions": 325,
    }
    with (Path(ledger) / "release-bound.json").open("x", encoding="utf-8") as handle:
        json.dump(ticket, handle, indent=2)
    files["release-ticket.json"] = (json.dumps(ticket, indent=2) + "\n").encode()
    manifest = {
        "schema": "sealed_narrative_test_manifest_v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {name: sha(raw) for name, raw in files.items()},
        "root_freeze_sha256": root_sha256,
        "package_parent": frozen["package_parent"],
        "gold_values_decoded": False,
    }
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    files["sealed-test-manifest.json"] = raw
    output.mkdir(parents=True)
    upload = output / "dataset-upload"
    upload.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    (upload / "LICENSE").write_bytes(license_file.read_bytes())
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Private one-shot full narrative evaluation",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    runner = (
        files["src/cloud/sealed_narrative_test_runner.py"]
        .decode()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(raw)),
            1,
        )
    )
    (output / "run.py").write_text(runner, encoding="utf-8", newline="\n")
    metadata = {
        "id": "aikyatansinha/portfolio-narrative-sealed-full325-v1",
        "title": "Portfolio Narrative Sealed Full325 V1",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [frozen["package_parent"]],
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "preflight.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha(raw),
                "runner_sha256": digest(output / "run.py"),
                "archive_sha256": digest(upload / "payload.zip"),
                "ticket_sha256": sha(files["release-ticket.json"]),
                "cohort_sha256": ticket["cohort_sha256"],
                "persistent_ledger": str(Path(ledger).resolve()),
                "actual_root_freeze_sha256": root_sha256,
                "gold_values_decoded": False,
            },
            indent=2,
        )
        + "\n"
    )
    return ticket


def dispatch(stage, ledger, push):
    """Validate the exact reviewed stage then consume the durable ticket once."""
    stage = Path(stage)
    preflight = json.loads((stage / "preflight.json").read_bytes())
    if (
        digest(stage / "run.py") != preflight["runner_sha256"]
        or digest(stage / "dataset-upload/payload.zip") != preflight["archive_sha256"]
    ):
        raise ValueError("Reviewed sealed stage changed before dispatch")
    with zipfile.ZipFile(stage / "dataset-upload/payload.zip") as archive:
        raw = archive.read("release-ticket.json")
        if sha(raw) != preflight["ticket_sha256"]:
            raise ValueError("Reviewed sealed ticket changed")
        ticket = json.loads(raw)
    return consume_dispatch(ledger, ticket, lambda: push(stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "metadata",
        "root-freeze",
        "root-sha256",
        "data",
        "ledger",
        "output",
        "dataset",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(
        args.source_root,
        args.metadata,
        args.root_freeze,
        args.root_sha256,
        args.data,
        args.ledger,
        args.output,
        args.dataset,
    )
