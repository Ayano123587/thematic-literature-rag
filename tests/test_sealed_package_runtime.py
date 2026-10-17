"""Match sealed runtime guards to the actual completed, reviewed package runtime."""

import ast
import importlib.metadata
import json
from pathlib import Path
import platform

import pytest

from src import sealed_narrative_test as sealed

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = (
    ROOT
    / "data/interim/narrative-production-package-stage-v2/publication/terminal-small-v1/narrative-production/package-execution.json"
)


def test_actual_completed_package_runtime_exact_four_keys(monkeypatch):
    if not EXECUTION.is_file():
        pytest.skip("Actual completed package metadata required")
    expected = json.loads(EXECUTION.read_bytes())["runtime"]
    assert set(expected) == {"python", "torch", "machine", "environment"}
    observed_host_machine = platform.machine()
    monkeypatch.setattr(platform, "python_version", lambda: expected["python"])
    monkeypatch.setattr(platform, "machine", lambda: expected["machine"])
    actual_versions = {"torch": expected["torch"], **expected["environment"]}
    monkeypatch.setattr(importlib.metadata, "version", actual_versions.__getitem__)
    assert sealed.runtime_now() == expected
    runner = ROOT / "src/cloud/sealed_narrative_test_runner.py"
    tree = ast.parse(runner.read_bytes())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "runtime_state"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "platform": platform,
        "importlib": importlib,
        "names": tuple(expected["environment"]),
    }
    exec(compile(ast.fix_missing_locations(module), str(runner), "exec"), namespace)
    assert namespace["runtime_state"]() == expected
    maps = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Dict)
        and {k.value for k in n.keys if isinstance(k, ast.Constant)} == set(expected)
    ]
    assert (
        len(maps) == 2
    )  # Before/after bootstrap and postflight use the same exact schema.
    for node in maps:
        machine = next(
            value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "machine"
        )
        assert ast.unparse(machine) == "platform.machine()"
    if observed_host_machine != expected["machine"]:
        monkeypatch.setattr(platform, "machine", lambda: observed_host_machine)
        assert sealed.runtime_now() != expected
        assert namespace["runtime_state"]() != expected
