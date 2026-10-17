"""Standard-library guards for an exploratory indexed-book cloud demonstration."""

import hashlib
import json
from pathlib import Path
import re
import zipfile

BOOKS = ("pride", "frankenstein")
PROMPTS = (
    "isolation rejection responsibility",
    "How are pride vanity and virtue distinguished from moral corruption?",
)
VALIDATION_IDS = "40d2ddd6b065469a8b947d84dbb630efcaac83acecc695606022116cc1a0f7cb"
TRANSPORT = {
    "src/indexed_demo_transport.py",
    "src/prepare_indexed_demo.py",
    "src/cloud/indexed_demo_runner.py",
}
BRIDGE = {
    "src/__init__.py",
    "src/context_assembly.py",
    "src/context_validation.py",
    "src/data/__init__.py",
    "src/data/prepare_fairytale.py",
    "src/features/__init__.py",
    "src/features/build_features.py",
    "src/full_generator_training.py",
    "src/full_reranker_training.py",
    "src/full_validation_selection.py",
    "src/indexed_narrative_cli.py",
    "src/joint_answer_selection.py",
    "src/models/__init__.py",
    "src/models/frozen_narrative_reader.py",
    "src/models/indexed_narrative_reader.py",
    "src/models/narrative_generator.py",
    "src/models/train_model.py",
    "src/narrative_training.py",
    "src/reranker_training.py",
    "src/resumable_training.py",
    "src/narrative_cli.py",
}
CORE_PACKAGES = {"transformers", "tokenizers", "numpy", "huggingface-hub"}
INDEX_PACKAGES = {"scikit-learn", "scipy", "joblib", "threadpoolctl"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def required_sha(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{64}", value)
        or value == "0" * 64
    ):
        raise ValueError("Actual externally frozen SHA256 required")
    return value


def at(root, name):
    root = Path(root).resolve()
    path = root / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Unsafe demo artifact")
    return path


def read(path):
    return json.loads(Path(path).read_bytes())


def authorize(metadata, freeze_path, freeze_sha):
    if digest(freeze_path) != required_sha(freeze_sha):
        raise ValueError("Root demo freeze differs")
    frozen = read(freeze_path)
    parent = frozen.get("package_parent", "")
    if (
        frozen.get("schema") != "root_frozen_indexed_demo_v1"
        or frozen.get("demo_authorized") is not True
        or frozen.get("joint_validation_questions") != 2372
        or frozen.get("joint_validation_ids_sha256") != VALIDATION_IDS
        or frozen.get("test_accessed") is not False
        or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*", parent)
    ):
        raise ValueError("Actual complete globally frozen package required")
    contract_path = at(metadata, "inference-contract.json")
    joint_path = at(metadata, "joint-status.json")
    policy_path = at(metadata, "joint-selection.json")
    if digest(contract_path) != required_sha(
        frozen["package_contract_sha256"]
    ) or digest(joint_path) != required_sha(frozen["joint_result_sha256"]):
        raise ValueError("Actual package/joint bytes differ")
    contract, joint, policy = read(contract_path), read(joint_path), read(policy_path)
    if (
        contract.get("schema") != "frozen_narrative_inference_v1"
        or contract.get("joint_result_sha256") != frozen["joint_result_sha256"]
        or contract.get("files", {}).get("joint-selection.json") != digest(policy_path)
        or joint.get("complete") is not True
        or joint.get("questions_completed") != 2372
        or joint.get("questions_required") != 2372
        or joint.get("test_accessed") is not False
        or joint.get("selection") != policy
        or policy.get("schema") != "joint_answer_selection_v1"
        or policy.get("validation_questions") != 2372
        or policy.get("validation_ids_sha256") != VALIDATION_IDS
        or policy.get("test_accessed") is not False
        or policy.get("selected_arm")
        not in {
            "bm25_standard",
            "bm25_windows",
            "reranked_standard",
            "reranked_windows",
        }
        or policy.get("selected_arm") != frozen.get("selected_arm")
    ):
        raise ValueError("Incomplete or inconsistent selected global policy")
    source = frozen.get("source_sha256", {})
    if set(source) != BRIDGE | TRANSPORT:
        raise ValueError("Complete expanded bridge source inventory required")
    for value in source.values():
        required_sha(value)
    if any(
        source.get("src/" + n) != v
        for n, v in contract.get("source_sha256", {}).items()
    ) or set(contract.get("source_sha256", {})) != {
        n.removeprefix("src/")
        for n in BRIDGE
        - {
            "src/indexed_narrative_cli.py",
            "src/models/indexed_narrative_reader.py",
            "src/models/train_model.py",
            "src/features/__init__.py",
            "src/features/build_features.py",
        }
    }:
        raise ValueError("Frozen inference and expanded bridge sources differ")
    runtime = frozen.get("runtime", {})
    if (
        set(runtime) != {"python", "torch", "environment"}
        or set(runtime.get("environment", {})) != CORE_PACKAGES
        or any(
            not isinstance(v, str) or not v
            for v in [
                runtime["python"],
                runtime["torch"],
                *runtime["environment"].values(),
            ]
        )
    ):
        raise ValueError("Actual complete inference/index runtime required")
    index_environment = frozen.get("index_environment", {})
    if (
        set(index_environment) != INDEX_PACKAGES
        or any(not isinstance(v, str) or not v for v in index_environment.values())
        or frozen.get("index_runtime_scope")
        != "prospective_demo_dependencies_not_recorded_model_training_runtime"
    ):
        raise ValueError("Explicit prospective index dependency pins required")
    if frozen.get("books") != list(BOOKS) or frozen.get("prompts") != list(PROMPTS):
        raise ValueError("Fixed illustrative prompts/books required")
    if set(frozen.get("assets", {})) != {
        "assets/index.lrag",
        *("assets/" + b + ".txt" for b in BOOKS),
    }:
        raise ValueError("Only existing index and processed requested books permitted")
    for value in frozen["assets"].values():
        required_sha(value)
    return frozen


def index_metadata(index, book_paths):
    # Opaque arrays are never opened or loaded on the host.
    with zipfile.ZipFile(index) as archive:
        if set(archive.namelist()) != {"metadata.json", "arrays.npz"}:
            raise ValueError("Exact existing numeric index archive required")
        info = archive.getinfo("metadata.json")
        if info.file_size > 50_000_000:
            raise ValueError("Oversized index metadata")
        metadata = json.loads(archive.read("metadata.json"))
    if metadata.get("schema") != 1 or metadata.get("backend") != "lsa":
        raise ValueError("Existing LSA hybrid demo index required")
    texts = {
        book: Path(path).read_text(encoding="utf-8")
        for book, path in book_paths.items()
    }
    passages = metadata.get("passages", [])
    if len({p["id"] for p in passages}) != len(passages):
        raise ValueError("Duplicate index passage identities")
    for book, text in texts.items():
        if not text.strip() or sha(text.encode()) != metadata.get(
            "source_hashes", {}
        ).get(book):
            raise ValueError("Processed book hash differs from index")
        selected = [p for p in passages if p["book"] == book]
        if not selected:
            raise ValueError("Requested indexed book is absent")
        for p in selected:
            start, end = p["start"], p["end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(text)
                or text[start:end] != p["text"]
                or p["id"] != f"{book}:{start}-{end}"
            ):
                raise ValueError("Indexed original source offsets differ")
    return metadata, texts


def authenticate_payload(payload, manifest, expected_manifest_sha):
    payload = Path(payload)
    if digest(payload / "indexed-demo-manifest.json") != expected_manifest_sha:
        raise ValueError("Demo manifest changed")
    expected = manifest["files"]
    allowed = (
        BRIDGE
        | TRANSPORT
        | {
            "root-freeze.json",
            "package-metadata/inference-contract.json",
            "package-metadata/joint-status.json",
            "package-metadata/joint-selection.json",
            "assets/index.lrag",
            *("assets/" + b + ".txt" for b in BOOKS),
        }
    )
    if set(expected) != allowed or manifest.get("schema") != "indexed_demo_manifest_v1":
        raise ValueError("Only exact demo sources/metadata/index/books permitted")
    actual = {
        p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()
    }
    if any(p.is_symlink() for p in payload.rglob("*")) or actual != set(expected) | {
        "indexed-demo-manifest.json"
    }:
        raise ValueError("Unexpected demo payload inventory")
    for name, value in expected.items():
        if digest(at(payload, name)) != required_sha(value):
            raise ValueError("Demo source/input bytes changed")


def authenticate_package(package, frozen, metadata):
    contract = read(at(metadata, "inference-contract.json"))
    files = contract["files"] | {
        "inference-contract.json": frozen["package_contract_sha256"]
    }
    package = Path(package)
    if package.is_symlink() or any(p.is_symlink() for p in package.rglob("*")):
        raise ValueError("Symlink package forbidden")
    if {
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    } != set(files):
        raise ValueError("Actual complete package inventory differs")
    for name, value in files.items():
        if digest(at(package, name)) != required_sha(value):
            raise ValueError("Actual package bytes differ before numerical imports")


def audit_outputs(
    results, metadata, texts, selected_arm, index_sha256, generator_sha256
):
    if len(results) != len(PROMPTS):
        raise ValueError("Incomplete fixed demonstration")
    passages = {p["id"]: p for p in metadata["passages"]}
    for question, result in zip(PROMPTS, results):
        if (
            result.get("question") != question
            or result.get("schema") != "exploratory-indexed-narrative-v1"
            or result.get("index_sha256") != index_sha256
            or result.get("retrieval_mode") != "hybrid"
            or result.get("candidate_limit_per_book") != 6
            or [b["book"] for b in result["books"]] != list(BOOKS)
            or not result.get("scope", "").startswith("Exploratory")
        ):
            raise ValueError("Exploratory demo identity differs")
        candidates = set()
        for output in result["books"]:
            book = output["book"]
            if output.get("abstained") is True:
                if output.get("reason") != "No retrieved evidence":
                    raise ValueError("Unknown abstention")
                continue
            rows = output["candidate_evidence"]
            if not 1 <= len(rows) <= 6 or len({r["citation"] for r in rows}) != len(
                rows
            ):
                raise ValueError("Candidate bounds differ")
            sections = {}
            for row in rows:
                p = passages.get(row["citation"])
                if (
                    p is None
                    or p["book"] != book
                    or any(row[k] != p[k] for k in ("start", "end", "text"))
                ):
                    raise ValueError("Candidate original offset replay differs")
                sections[row["citation"]] = p
                candidates.add(row["citation"])
            answer = output["result"]
            if (
                answer.get("story") != book
                or answer.get("source_sha256") != sha(texts[book].encode())
                or answer.get("selected_arm") != selected_arm
                or answer.get("generator_weights_sha256") != generator_sha256
                or not answer.get("retrieved_evidence")
            ):
                raise ValueError("Reader output source/policy differs")
            for c in answer["retrieved_evidence"]:
                p = sections.get(c.get("section"))
                a, b = c.get("start"), c.get("end")
                if (
                    p is None
                    or type(a) is not int
                    or type(b) is not int
                    or not p["start"] <= a < b <= p["end"]
                    or texts[book][a:b] != c.get("text")
                ):
                    raise ValueError(
                        "Citation replay differs from selected section/full source"
                    )
        import math

        for link in result["cross_book_connections"]:
            a, b = link["left"], link["right"]
            score = link["cosine_similarity"]
            if (
                a not in candidates
                or b not in candidates
                or passages[a]["book"] == passages[b]["book"]
                or not math.isfinite(score)
                or not -1.00000001 <= score <= 1.00000001
            ):
                raise ValueError("Cross-book link replay differs")
