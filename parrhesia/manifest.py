"""
Run manifest system for reproducibility tracking.

Each run gets a structured YAML manifest at runs/<run-id>/manifest.yaml
that captures every command, hyperparameter, input, output, and metric.

Usage:
    from parrhesia.manifest import create_manifest, record_step, load_manifest

    manifest = create_manifest("run-001-A-qwen3-8b", "A", "Qwen/Qwen3-8B", "First OCT run")
    record_step("run-001-A-qwen3-8b", "prompts", "python -m parrhesia.data.generate_oct prompts ...",
                hyperparameters={"num_per_category": 50}, outputs={"prompts": "data/generated/oct/prompts.jsonl"})
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


@dataclass
class StepRecord:
    step_name: str
    command: str
    started_at: str = ""
    completed_at: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    hub_refs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RunManifest:
    run_id: str
    approach: str
    base_model: str
    created_at: str
    git_sha: str
    git_dirty: bool
    description: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    hub_datasets: dict[str, str] = field(default_factory=dict)
    hub_models: dict[str, str] = field(default_factory=dict)


def generate_run_id(approach: str, base_model: str) -> str:
    """Generate the next run ID by scanning runs/ for the max run-NNN-* and incrementing.

    Format: run-NNN-<approach>-<model_short>
    Example: run-001-A-qwen3-8b
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Find the highest existing run number
    max_num = 0
    for entry in RUNS_DIR.iterdir():
        if entry.is_dir():
            match = re.match(r"run-(\d+)-", entry.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

    next_num = max_num + 1

    # Shorten model name: "Qwen/Qwen3-8B" -> "qwen3-8b"
    model_short = base_model.split("/")[-1].lower()

    return f"run-{next_num:03d}-{approach}-{model_short}"


def get_git_info() -> tuple[str, bool]:
    """Return (sha, is_dirty) via subprocess. Returns empty sha if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", False


def get_environment_info() -> dict[str, str]:
    """Capture python, pytorch, key library versions, CUDA, and GPU info."""
    env = {}

    import sys
    env["python"] = sys.version.split()[0]

    try:
        import torch
        env["pytorch"] = str(torch.__version__)
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda or ""
            env["gpu"] = torch.cuda.get_device_name(0)
    except (ImportError, OSError):
        pass

    for pkg in ["unsloth", "peft", "trl", "transformers", "datasets"]:
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except (ImportError, NotImplementedError, OSError):
            pass

    return env


def create_manifest(
    run_id: str,
    approach: str,
    base_model: str,
    description: str = "",
) -> RunManifest:
    """Create a new manifest and its runs/<id>/ directory, then save to YAML."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git_sha, git_dirty = get_git_info()
    env = get_environment_info()

    manifest = RunManifest(
        run_id=run_id,
        approach=approach,
        base_model=base_model,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        description=description,
        environment=env,
    )

    save_manifest(manifest)
    return manifest


def _manifest_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "manifest.yaml"


def load_manifest(run_id: str) -> RunManifest:
    """Load a manifest from runs/<run-id>/manifest.yaml."""
    path = _manifest_path(run_id)
    with open(path) as f:
        data = yaml.safe_load(f)

    return RunManifest(
        run_id=data["run_id"],
        approach=data["approach"],
        base_model=data["base_model"],
        created_at=data["created_at"],
        git_sha=data.get("git_sha", ""),
        git_dirty=data.get("git_dirty", False),
        description=data.get("description", ""),
        environment=data.get("environment", {}),
        steps=data.get("steps", []),
        hub_datasets=data.get("hub_datasets", {}),
        hub_models=data.get("hub_models", {}),
    )


def save_manifest(manifest: RunManifest) -> Path:
    """Save a manifest to runs/<run-id>/manifest.yaml."""
    path = _manifest_path(manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "run_id": manifest.run_id,
        "approach": manifest.approach,
        "base_model": manifest.base_model,
        "created_at": manifest.created_at,
        "git_sha": manifest.git_sha,
        "git_dirty": manifest.git_dirty,
        "description": manifest.description,
        "environment": manifest.environment,
        "steps": manifest.steps,
        "hub_datasets": manifest.hub_datasets,
        "hub_models": manifest.hub_models,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

    return path


def record_step(
    run_id: str,
    step_name: str,
    command: str,
    hyperparameters: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    hub_refs: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> None:
    """Append a step record to the manifest for the given run.

    No-ops with a warning if the manifest is missing. This happens when a step
    runs somewhere that only has a fresh clone (e.g. a RunPod pod cloning the
    repo, where the run's manifest was created locally and never pushed). A
    missing manifest should never crash the pipeline step that is recording
    after its real work (training, eval) has already completed and been saved.
    """
    try:
        manifest = load_manifest(run_id)
    except FileNotFoundError:
        print(
            f"[manifest] runs/{run_id}/manifest.yaml not found; "
            f"skipping step record for '{step_name}'."
        )
        return

    step = {
        "step_name": step_name,
        "command": command,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters or {},
        "inputs": inputs or {},
        "outputs": outputs or {},
        "hub_refs": hub_refs or {},
        "metrics": metrics or {},
        "notes": notes,
    }

    manifest.steps.append(step)
    save_manifest(manifest)


def get_current_run_id() -> str | None:
    """Read the current run ID from the PARRHESIA_RUN_ID env var."""
    return os.environ.get("PARRHESIA_RUN_ID")


def reconstruct_commands(run_id: str) -> list[str]:
    """Return an ordered list of shell commands from the manifest steps."""
    manifest = load_manifest(run_id)
    return [step["command"] for step in manifest.steps if step.get("command")]


def format_manifest_markdown(manifest: RunManifest) -> str:
    """Render a manifest as markdown for display, cards, and log.md."""
    lines = []
    lines.append(f"# {manifest.run_id}")
    lines.append("")
    lines.append(f"**Approach:** {manifest.approach}")
    lines.append(f"**Base model:** {manifest.base_model}")
    lines.append(f"**Created:** {manifest.created_at}")
    if manifest.description:
        lines.append(f"**Description:** {manifest.description}")
    lines.append(f"**Git SHA:** {manifest.git_sha}" + (" (dirty)" if manifest.git_dirty else ""))
    lines.append("")

    if manifest.environment:
        lines.append("## Environment")
        lines.append("")
        lines.append("| Component | Version |")
        lines.append("|-----------|---------|")
        for k, v in manifest.environment.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if manifest.steps:
        lines.append("## Steps")
        lines.append("")
        for i, step in enumerate(manifest.steps, 1):
            lines.append(f"### Step {i}: {step['step_name']}")
            lines.append("")
            lines.append(f"```bash\n{step['command']}\n```")
            lines.append("")

            if step.get("hyperparameters"):
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                for k, v in step["hyperparameters"].items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")

            if step.get("inputs"):
                lines.append("**Inputs:**")
                for k, v in step["inputs"].items():
                    lines.append(f"- {k}: `{v}`")
                lines.append("")

            if step.get("outputs"):
                lines.append("**Outputs:**")
                for k, v in step["outputs"].items():
                    lines.append(f"- {k}: `{v}`")
                lines.append("")

            if step.get("metrics"):
                lines.append("**Metrics:**")
                for k, v in step["metrics"].items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

            if step.get("hub_refs"):
                lines.append("**Hub:**")
                for k, v in step["hub_refs"].items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

            if step.get("notes"):
                lines.append(f"**Notes:** {step['notes']}")
                lines.append("")

    if manifest.hub_datasets:
        lines.append("## Hub Datasets")
        lines.append("")
        for k, v in manifest.hub_datasets.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if manifest.hub_models:
        lines.append("## Hub Models")
        lines.append("")
        for k, v in manifest.hub_models.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    return "\n".join(lines)
