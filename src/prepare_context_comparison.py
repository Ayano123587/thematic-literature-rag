"""Prepare a frozen, complete paired context comparison without model loading."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

SCIENCE = {
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/prepare_fairytale.py",
    "src/narrative_training.py",
    "src/full_generator_training.py",
    "src/resumable_training.py",
    "src/full_validation_selection.py",
    "src/reranker_training.py",
    "src/context_assembly.py",
    "src/context_validation.py",
    "docs/full-coverage-protocol.json",
    "docs/context-assembly-protocol.json",
}
NEW_SOURCE = {
    "src/prepare_context_comparison.py",
    "src/cloud/context_comparison_runner.py",
}
MANIFEST = "721733169b0c1e7b7c1d07fe1880d3c871d72017be6fc61923be126feea73225"
IDS = "40d2ddd6b065469a8b947d84dbb630efcaac83acecc695606022116cc1a0f7cb"
GENERATOR_PROTOCOL = "5e4fc152ac55e2153b18885d88be380987a74ddae1202c2c5ef10be5a8c1fa88"
TRAINING_STATUS = "ea131acfc1aa1f515b9bdff386aa63ab45d633f6d8bbf9a5c0eea90d4dc4709a"
TRAINING_PROVENANCE = "83e3d84e7fe3533064924fb6812c98fe61dd5943f317cd51aab459b27a740b1d"
WEIGHTS = [
    "881ecd3d2e20cfb1ddf96ac598ad90a610007dfa4ff04cfc024e4afd118bc686",
    "6752700e682cf3f25f4c50eef2ac44b7d8c186fe3497632fae97b264c9d819ac",
    "4b94b6c580d1aa182b7b04cbeede8f3430bc1b019709266637f973b51137d227",
    "4a51a48953e6c02c2c8aa5325a02600ae7a65dc8080abb5fb45e2b7131c39538",
]
SHARDS = [f"shard-{i:05d}.json" for i in range(0, 2372, 64)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def digest_value(value):
    return (
        isinstance(value, str)
        and re.fullmatch("[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def kernel(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[\w-]+/[\w-]+/[1-9][0-9]*", value) is not None
    )


def authenticate(path, expected, selection, root=Path(".")):
    """Unknown selection identities and placeholder contracts never release QA."""
    if not digest_value(expected) or sha(path) != expected:
        raise ValueError("Exact frozen comparison inputs SHA256 required")
    contract = read(path)
    keys = {
        "schema",
        "selection_sha256",
        "selected_epoch",
        "weights_sha256",
        "checkpoint_receipt_sha256",
        "baseline_status_sha256",
        "baseline_identity_sha256",
        "baseline_shards_sha256",
        "source_manifest_sha256",
        "validation_ids_sha256",
        "generator_protocol_sha256",
        "context_protocol_sha256",
        "source_sha256",
        "runtime",
        "training_kernel",
        "selection_kernel",
        "baseline_kernel",
        "training_status_sha256",
        "training_provenance_sha256",
        "resume",
    }
    if set(contract) != keys or contract["schema"] != "context_comparison_inputs_v1":
        raise ValueError("Complete frozen comparison contract required")
    epoch = contract["selected_epoch"]
    if (
        type(epoch) is not int
        or not 0 <= epoch <= 3
        or contract["weights_sha256"] != WEIGHTS[epoch]
    ):
        raise ValueError("Actual selected complete training snapshot required")
    hashes = (
        "selection_sha256",
        "checkpoint_receipt_sha256",
        "baseline_status_sha256",
        "baseline_identity_sha256",
        "context_protocol_sha256",
    )
    if (
        any(not digest_value(contract[k]) for k in hashes)
        or contract["source_manifest_sha256"] != MANIFEST
        or contract["validation_ids_sha256"] != IDS
        or contract["generator_protocol_sha256"] != GENERATOR_PROTOCOL
        or contract["training_status_sha256"] != TRAINING_STATUS
        or contract["training_provenance_sha256"] != TRAINING_PROVENANCE
    ):
        raise ValueError(
            "Frozen full source, protocol and artifact identities required"
        )
    shards = contract["baseline_shards_sha256"]
    if (
        not isinstance(shards, dict)
        or set(shards) != set(SHARDS)
        or any(not digest_value(v) for v in shards.values())
    ):
        raise ValueError("All 38 selected-baseline shard hashes required")
    if (
        any(
            not kernel(contract[k])
            for k in ("training_kernel", "selection_kernel", "baseline_kernel")
        )
        or contract["training_kernel"]
        != "aikyatansinha/portfolio-narrative-full-resume-train-v6/1"
    ):
        raise ValueError("Explicit immutable comparison kernel versions required")
    sources = contract["source_sha256"]
    if (
        not isinstance(sources, dict)
        or set(sources) != SCIENCE
        or any(
            not digest_value(v) or sha(Path(root) / n) != v for n, v in sources.items()
        )
    ):
        raise ValueError("Prespecified scientific source bytes changed")
    if (
        sources["docs/full-coverage-protocol.json"] != GENERATOR_PROTOCOL
        or sources["docs/context-assembly-protocol.json"]
        != contract["context_protocol_sha256"]
    ):
        raise ValueError("Prespecified protocol byte identity changed")
    if sha(selection) != contract["selection_sha256"]:
        raise ValueError("Actual audited selection artifact required")
    chosen = read(selection)
    candidates = chosen.get("candidates", [])
    if (
        chosen.get("status") != "validation_checkpoint_selection"
        or chosen.get("task") != "generator"
        or chosen.get("validation_questions") != 2372
        or chosen.get("validation_ids_sha256") != IDS
        or chosen.get("manifest_sha256") != MANIFEST
        or chosen.get("protocol_sha256") != GENERATOR_PROTOCOL
        or chosen.get("audit_source_sha256")
        != sources["src/full_validation_selection.py"]
        or chosen.get("test_accessed") is not False
        or chosen.get("final_test_authorized") is not False
        or [r.get("epoch") for r in candidates] != list(range(4))
    ):
        raise ValueError("Complete initial-through-final checkpoint selection required")
    for i, row in enumerate(candidates):
        value = row.get("validation", {}).get("f1")
        if (
            row.get("weights_sha256") != WEIGHTS[i]
            or row.get("validation", {}).get("n") != 2372
            or type(value) not in (int, float)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "Complete finite full-validation candidate metrics required"
            )
    if (
        chosen.get("selected") != max(candidates, key=lambda r: r["validation"]["f1"])
        or chosen["selected"]["epoch"] != epoch
    ):
        raise ValueError("Checkpoint differs from fixed F1 and earliest-tie selection")
    receipts = chosen.get("artifact_receipts", [])
    if (
        [r.get("epoch") for r in receipts] != list(range(4))
        or receipts[epoch].get("checkpoint_receipt_sha256")
        != contract["checkpoint_receipt_sha256"]
        or receipts[epoch].get("validation_status_sha256")
        != contract["baseline_status_sha256"]
    ):
        raise ValueError("Selected checkpoint or baseline receipt changed")
    lineage = chosen.get("training_identity", {})
    runtime = {k: lineage.get(k) for k in ("python", "torch", "machine", "environment")}
    versions = runtime["environment"]
    if (
        contract["runtime"] != runtime
        or any(
            not isinstance(runtime[k], str) or not runtime[k]
            for k in ("python", "torch", "machine")
        )
        or not isinstance(versions, dict)
        or set(versions) != {"transformers", "tokenizers", "numpy", "huggingface-hub"}
        or any(not isinstance(v, str) or not v for v in versions.values())
    ):
        raise ValueError("Actual selected generator runtime required")
    if (
        lineage.get("source_manifest_sha256") != MANIFEST
        or lineage.get("protocol_sha256") != GENERATOR_PROTOCOL
        or lineage.get("smoke") is not False
        or lineage.get("micro_batch") != 2
        or lineage.get("threads") != 4
        or lineage.get("generator")
        != read(Path(root) / "docs/full-coverage-protocol.json")["generator"]
        or lineage.get("source_sha256")
        != {
            n: sources["src/" + n]
            for n in (
                "full_generator_training.py",
                "resumable_training.py",
                "narrative_training.py",
                "data/prepare_fairytale.py",
            )
        }
    ):
        raise ValueError("Actual completed generator training identity required")
    validate_resume_contract(contract["resume"])
    return contract, chosen


def validate_resume_contract(resume):
    if resume is None:
        return
    keys = {
        "kernel",
        "identity_sha256",
        "status_sha256",
        "retrieval_plan_sha256",
        "shards_sha256",
    }
    if (
        not isinstance(resume, dict)
        or set(resume) != keys
        or not kernel(resume["kernel"])
        or any(
            not digest_value(resume[k])
            for k in ("identity_sha256", "status_sha256", "retrieval_plan_sha256")
        )
    ):
        raise ValueError("Exact externally authenticated continuation required")
    shards = resume["shards_sha256"]
    if (
        not isinstance(shards, dict)
        or not 0 < len(shards) < len(SHARDS)
        or set(shards) != set(SHARDS[: len(shards)])
        or any(not digest_value(v) for v in shards.values())
    ):
        raise ValueError("Exact completed paired-shard prefix required")


def comparison_identity(contract, selection):
    return {
        "training": selection["training_identity"],
        "selected_epoch": contract["selected_epoch"],
        "weights_sha256": contract["weights_sha256"],
        "selection_sha256": contract["selection_sha256"],
        "baseline_status_sha256": contract["baseline_status_sha256"],
        "context_protocol_sha256": contract["context_protocol_sha256"],
        "validation_ids_sha256": IDS,
        "shard_size": 64,
        "batch_size": 8,
        "source_sha256": {
            name: contract["source_sha256"]["src/" + name]
            for name in (
                "context_assembly.py",
                "context_validation.py",
                "full_validation_selection.py",
            )
        },
    }


def authenticate_resume(directory, resume, identity=None):
    if resume is None:
        if directory is not None:
            raise ValueError("Unfrozen continuation directory forbidden")
        return
    if directory is None:
        raise ValueError("Frozen continuation directory required")
    directory = Path(directory)
    expected = {
        "identity.json": resume["identity_sha256"],
        "comparison-status.json": resume["status_sha256"],
        "retrieval-plan.json": resume["retrieval_plan_sha256"],
        **resume["shards_sha256"],
    }
    if any(sha(directory / n) != digest for n, digest in expected.items()) or {
        p.name for p in directory.glob("shard-*.json")
    } != set(resume["shards_sha256"]):
        raise ValueError("Continuation plan, status or paired shard bytes changed")
    status = read(directory / "comparison-status.json")
    if (
        status.get("complete") is not False
        or status.get("questions_required") != 2372
        or status.get("questions_completed") != 64 * len(resume["shards_sha256"])
        or status.get("test_accessed") is not False
        or status.get("final_test_authorized") is not False
        or any(
            k in status
            for k in ("baseline", "candidate", "decision", "story_paired_F1_interval")
        )
    ):
        raise ValueError("Only complete partial shards without promotion can continue")
    saved_identity = read(directory / "identity.json")
    retrieval = read(directory / "retrieval-plan.json")
    if (
        identity is None
        or saved_identity != identity
        or status.get("identity") != identity
        or retrieval.get("identity") != identity
    ):
        raise ValueError(
            "Continuation scientific identity differs before model loading"
        )
    queries = retrieval.get("queries", [])
    if (
        len(queries) != 2372
        or len({r["id"] for r in queries}) != 2372
        or hashlib.sha256(json.dumps([r["id"] for r in queries]).encode()).hexdigest()
        != IDS
        or hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest()
        != retrieval.get("queries_sha256")
    ):
        raise ValueError("Frozen question-only retrieval coverage changed")
    for index, name in enumerate(SHARDS[: len(resume["shards_sha256"])]):
        shard = read(directory / name)
        pairs = shard.get("pairs", [])
        if (
            shard.get("identity") != identity
            or [r["id"] for r in pairs]
            != [r["id"] for r in queries[index * 64 : (index + 1) * 64]]
            or hashlib.sha256(
                json.dumps(pairs, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            != shard.get("pairs_sha256")
        ):
            raise ValueError("Externally bound paired continuation shard differs")


def prepare(
    data,
    frozen_inputs,
    frozen_inputs_sha256,
    selection,
    output,
    dataset,
    kernel_id,
    previous=None,
    license_file=None,
):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh context-comparison staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed transport before private data reads")
    contract, selected = authenticate(frozen_inputs, frozen_inputs_sha256, selection)
    authenticate_resume(
        previous, contract["resume"], comparison_identity(contract, selected)
    )
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[\w-]+/[\w-]+", v)
        for v in (dataset, kernel_id)
    ):
        raise ValueError("Explicit owner/slug identities required")
    data = Path(data)
    if sha(data / "manifest.json") != MANIFEST:
        raise ValueError("Complete fixed source dataset required")
    from src.context_validation import check_protocol
    from src.full_validation_selection import validation_cohort
    from src.narrative_training import load_split

    check_protocol("docs/context-assembly-protocol.json")
    validation_cohort(data, read("docs/full-coverage-protocol.json"))
    if len(load_split(data, "train")[1]) != 7883:
        raise ValueError("Complete original training source payload required")
    files = {n: Path(n).read_bytes() for n in sorted(SCIENCE | NEW_SOURCE)}
    for path in sorted(data.rglob("*")):
        relative = path.relative_to(data)
        if path.is_file() and (
            relative.as_posix() == "manifest.json"
            or relative.parts[0] in ("train", "validation")
        ):
            files["private-data/" + relative.as_posix()] = path.read_bytes()
    files["frozen-comparison-inputs.json"] = Path(frozen_inputs).read_bytes()
    files["selected-generator.json"] = Path(selection).read_bytes()
    if license_file is None:
        matches = list(
            Path("data/external/fairytaleqa").glob("FairytaleQAData-*/LICENSE")
        )
        if len(matches) != 1:
            raise ValueError("Explicit upstream --license-file required")
        license_file = matches[0]
    notice = Path(license_file).read_bytes()
    if not notice.strip():
        raise ValueError("Upstream dataset license notice must be preserved")
    files["NOTICE/FairytaleQA-LICENSE"] = notice
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    (upload / "LICENSE").write_bytes(notice)
    rows = []
    for name, raw in files.items():
        flat = name.replace("/", "__")
        (upload / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "context_comparison_transport_v1",
        "dispatchable": True,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "frozen_inputs_sha256": frozen_inputs_sha256,
        "files": rows,
        "test_QA_packaged": False,
        "weights_refitted": False,
        "license_notice_sha256": hashlib.sha256(notice).hexdigest(),
    }
    path = upload / "context-source-manifest.json"
    path.write_bytes(encoded(manifest))
    template = files["src/cloud/context_comparison_runner.py"].decode()
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One explicit runner manifest binding required")
    (output / "run.py").write_text(
        template.replace(token, "EXPECTED_MANIFEST_SHA256 = " + repr(sha(path))),
        encoding="utf-8",
        newline="\n",
    )
    (upload / "dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": dataset,
                "title": dataset.split("/")[1].replace("-", " "),
                "licenses": [{"name": "other"}],
            }
        )
    )
    mounted = [
        contract[k] for k in ("training_kernel", "selection_kernel", "baseline_kernel")
    ] + ([contract["resume"]["kernel"]] if contract["resume"] else [])
    if len(set(mounted)) != len(mounted):
        raise ValueError("Each immutable artifact kernel must be mounted once")
    (output / "kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[1].replace("-", " "),
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "kernel_sources": mounted,
            }
        )
    )
    (output / "preflight.json").write_bytes(
        encoded(
            {
                "source_commit": manifest["source_commit"],
                "source_manifest_sha256": sha(path),
                "runner_sha256": sha(output / "run.py"),
                "license_notice_sha256": manifest["license_notice_sha256"],
                "test_QA_packaged": False,
                "stage_only_not_dispatched": True,
            }
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "frozen-inputs",
        "frozen-inputs-sha256",
        "selection",
        "output",
        "dataset",
        "kernel-id",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous")
    parser.add_argument("--license-file")
    prepare(**vars(parser.parse_args()))
