"""Load and normalize run manifests, handling inconsistencies across runs 1-5."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_PREFIX = "/workspace/parrhesia/"


def _strip_workspace(path: str) -> str:
    """Strip absolute /workspace/parrhesia/ prefix, make relative."""
    if path.startswith(WORKSPACE_PREFIX):
        return path[len(WORKSPACE_PREFIX):]
    return path


def _normalize_paths_in_dict(d: dict[str, str]) -> dict[str, str]:
    return {k: _strip_workspace(v) for k, v in d.items()}


@dataclass
class Step:
    index: int
    name: str
    command: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    completed_at: str = ""
    hub_refs: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"[{self.index}] {self.name}"

    def viewable_files(self) -> list[tuple[str, str]]:
        """Return (label, path) for files that can be opened in the viewer."""
        files = []
        for label, path in {**self.inputs, **self.outputs}.items():
            if path.endswith((".jsonl", ".json", ".yaml", ".yml", ".md")):
                files.append((label, path))
        return files


@dataclass
class RunManifest:
    run_id: str
    approach: str
    base_model: str
    created_at: str
    description: str
    environment: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    hub_datasets: dict[str, str] = field(default_factory=dict)
    hub_models: dict[str, str] = field(default_factory=dict)
    git_sha: str = ""
    git_dirty: bool = False
    manifest_path: str = ""

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def short_description(self, max_len: int = 60) -> str:
        d = self.description
        if len(d) > max_len:
            return d[: max_len - 1] + "\u2026"
        return d

    def result_files(self, project_root: Path) -> list[str]:
        """Discover result files in results/<run-id>/ for runs with no steps."""
        results_dir = project_root / "results" / self.run_id
        if not results_dir.is_dir():
            return []
        found = []
        for root, _dirs, files in os.walk(results_dir):
            for f in sorted(files):
                if f.endswith((".json", ".jsonl", ".yaml", ".yml")):
                    rel = os.path.relpath(os.path.join(root, f), project_root)
                    found.append(rel)
        return found


def _parse_step(index: int, raw: dict[str, Any]) -> Step:
    """Normalize a single step dict, handling field name variations."""
    # Run 004 uses 'name' instead of 'step_name'
    name = raw.get("step_name") or raw.get("name") or f"step_{index}"
    # Run 004 uses 'description' instead of 'command'
    command = raw.get("command") or raw.get("description") or ""

    inputs = _normalize_paths_in_dict(raw.get("inputs") or {})
    outputs = _normalize_paths_in_dict(raw.get("outputs") or {})

    # Collect extra fields not in our known set
    known_keys = {
        "step_name", "name", "command", "description", "hyperparameters",
        "inputs", "outputs", "metrics", "notes", "completed_at", "hub_refs",
    }
    extra = {k: v for k, v in raw.items() if k not in known_keys}

    return Step(
        index=index,
        name=name,
        command=command,
        hyperparameters=raw.get("hyperparameters") or {},
        inputs=inputs,
        outputs=outputs,
        metrics=raw.get("metrics") or {},
        notes=str(raw.get("notes") or ""),
        completed_at=str(raw.get("completed_at") or ""),
        hub_refs=raw.get("hub_refs") or {},
        extra=extra,
    )


def load_manifest(path: str | Path) -> RunManifest:
    """Load a single manifest file and normalize it."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    steps = []
    for i, step_raw in enumerate(raw.get("steps") or []):
        steps.append(_parse_step(i, step_raw))

    return RunManifest(
        run_id=raw.get("run_id", path.parent.name),
        approach=raw.get("approach", "?"),
        base_model=raw.get("base_model", ""),
        created_at=str(raw.get("created_at", "")),
        description=raw.get("description", ""),
        environment=raw.get("environment") or {},
        steps=steps,
        results=raw.get("results") or {},
        hub_datasets=raw.get("hub_datasets") or {},
        hub_models=raw.get("hub_models") or {},
        git_sha=str(raw.get("git_sha", "")),
        git_dirty=bool(raw.get("git_dirty", False)),
        manifest_path=str(path),
    )


def discover_runs(project_root: str | Path) -> list[RunManifest]:
    """Find all runs/*/manifest.yaml and return loaded manifests, newest first."""
    project_root = Path(project_root)
    runs_dir = project_root / "runs"
    if not runs_dir.is_dir():
        return []

    manifests = []
    for run_dir in sorted(runs_dir.iterdir()):
        manifest_path = run_dir / "manifest.yaml"
        if manifest_path.is_file():
            try:
                manifests.append(load_manifest(manifest_path))
            except Exception as e:
                print(f"Warning: failed to load {manifest_path}: {e}")

    # Sort newest first by created_at
    manifests.sort(key=lambda m: m.created_at, reverse=True)
    return manifests
