"""
HuggingFace Hub integration for parrhesia.

All repos are private under the `daios` org. Versioning uses branches named
after run IDs. Each push includes auto-generated dataset/model cards.

Usage:
    from parrhesia.hub import push_dataset, pull_dataset, push_adapter, pull_adapter

    push_dataset("data/generated/oct/dpo_pairs.jsonl", "daios/parrhesia-oct-data",
                 run_id="run-001-A-qwen3-8b", split_name="dpo_pairs")
    pull_adapter("daios/parrhesia-oct-8b", "adapters/parrhesia-oct-8b",
                 run_id="run-001-A-qwen3-8b")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HUB_ORG = "daios"

DATASET_REPOS = {
    "oct": f"{HUB_ORG}/parrhesia-oct-data",
    "introspection": f"{HUB_ORG}/parrhesia-introspection-data",
    "sft": f"{HUB_ORG}/parrhesia-sft-data",
    "dpo": f"{HUB_ORG}/parrhesia-dpo-data",
    "benchmark_scenarios": f"{HUB_ORG}/parrhesia-benchmark-scenarios",
}

MODEL_REPOS = {
    "oct-8b": f"{HUB_ORG}/parrhesia-oct-8b",
    "oct-introspect-8b": f"{HUB_ORG}/parrhesia-oct-introspect-8b",
    "oct-14b": f"{HUB_ORG}/parrhesia-oct-14b",
}

EVAL_REPO = f"{HUB_ORG}/parrhesia-eval-results"


def derive_repo_id(local_path: str | Path) -> str:
    """Derive a Hub repo id from a local adapter directory.

    Convention: ``adapters/<NAME>`` → ``<HUB_ORG>/<NAME>``. Falls back to a
    direct lookup in ``MODEL_REPOS`` if a matching key exists, so legacy
    runs that pushed to specifically-named repos still resolve correctly.
    """
    name = Path(local_path).name
    # Allow legacy lookups (e.g. "oct-introspect-8b") to resolve via the table.
    if name in MODEL_REPOS:
        return MODEL_REPOS[name]
    return f"{HUB_ORG}/{name}"


def _get_api():
    """Get an authenticated HfApi instance."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token)


def ensure_repo_exists(repo_id: str, repo_type: str = "dataset", private: bool = True) -> str:
    """Create a Hub repo if it doesn't exist. Returns the repo URL."""
    api = _get_api()
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    return str(repo_url)


def _create_branch(repo_id: str, branch: str, repo_type: str = "dataset") -> None:
    """Create a branch on a Hub repo. Silently ignores if branch already exists."""
    api = _get_api()
    try:
        api.create_branch(repo_id=repo_id, branch=branch, repo_type=repo_type)
    except Exception:
        # Branch already exists (409) or other error — safe to ignore
        pass


def push_dataset(
    data_path: str | Path,
    repo_id: str,
    run_id: str,
    split_name: str = "train",
    card_metadata: dict | None = None,
) -> str:
    """Upload a JSONL dataset file + dataset card to Hub on a run-ID branch.

    Returns the Hub URL for the dataset.
    """
    from huggingface_hub import HfApi

    data_path = Path(data_path)
    api = _get_api()

    ensure_repo_exists(repo_id, repo_type="dataset")
    _create_branch(repo_id, run_id, repo_type="dataset")

    # Upload the data file
    api.upload_file(
        path_or_fileobj=str(data_path),
        path_in_repo=f"data/{split_name}.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        revision=run_id,
        commit_message=f"Upload {split_name} from {run_id}",
    )

    # Generate and upload dataset card
    card_content = generate_dataset_card(repo_id, run_id, card_metadata or {})
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=run_id,
        commit_message=f"Update dataset card for {run_id}",
    )

    return f"https://huggingface.co/datasets/{repo_id}/tree/{run_id}"


def pull_dataset(
    repo_id: str,
    output_path: str | Path,
    run_id: str | None = None,
    split_name: str = "train",
) -> Path:
    """Download a dataset file from Hub. Returns the local path."""
    from huggingface_hub import hf_hub_download

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=f"data/{split_name}.jsonl",
        repo_type="dataset",
        revision=run_id or "main",
        local_dir=str(output_path.parent),
        token=token,
    )

    return Path(downloaded)


def push_adapter(
    adapter_dir: str | Path,
    repo_id: str,
    run_id: str,
    card_metadata: dict | None = None,
) -> str:
    """Upload a LoRA adapter directory to Hub on a run-ID branch.

    Excludes checkpoint-* directories to save space.
    Returns the Hub URL.
    """
    adapter_dir = Path(adapter_dir)
    api = _get_api()

    ensure_repo_exists(repo_id, repo_type="model")
    _create_branch(repo_id, run_id, repo_type="model")

    # Upload adapter directory, excluding checkpoints
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=run_id,
        commit_message=f"Upload adapter from {run_id}",
        ignore_patterns=["checkpoint-*", "*.pt", "optimizer.pt"],
    )

    # Generate and upload model card
    card_content = generate_model_card(repo_id, run_id, card_metadata or {})
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        revision=run_id,
        commit_message=f"Update model card for {run_id}",
    )

    return f"https://huggingface.co/datasets/{repo_id}/tree/{run_id}"


def pull_adapter(
    repo_id: str,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Download a LoRA adapter from Hub. Returns the local directory path."""
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=run_id or "main",
        local_dir=str(output_dir),
        token=token,
    )

    return Path(downloaded)


def push_eval_results(
    results_path: str | Path,
    run_id: str,
    model_name: str,
) -> str:
    """Upload evaluation results to the eval results repo on a run-ID branch."""
    results_path = Path(results_path)
    api = _get_api()

    ensure_repo_exists(EVAL_REPO, repo_type="dataset")
    _create_branch(EVAL_REPO, run_id, repo_type="dataset")

    # Clean the model name for use as filename
    safe_name = model_name.replace("/", "-").replace(" ", "-")

    api.upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo=f"results/{safe_name}.json",
        repo_id=EVAL_REPO,
        repo_type="dataset",
        revision=run_id,
        commit_message=f"Upload {safe_name} eval results from {run_id}",
    )

    return f"https://huggingface.co/datasets/{EVAL_REPO}/tree/{run_id}"


def generate_dataset_card(repo_id: str, run_id: str, metadata: dict) -> str:
    """Generate a HuggingFace dataset card with run metadata."""
    # Extract info from metadata
    description = metadata.get("description", "")
    steps = metadata.get("steps", [])
    base_model = metadata.get("base_model", "")
    approach = metadata.get("approach", "")

    # Build reproduction commands
    commands = ""
    if steps:
        commands = "\n".join(f"```bash\n{s['command']}\n```" for s in steps if s.get("command"))

    card = f"""\
---
license: apache-2.0
tags:
  - parrhesia
  - sycophancy
  - character-training
  - {run_id}
---

# {repo_id.split('/')[-1]}

Part of the [parrhesia](https://github.com/daiostech/parrhesia) project — Aristotelian virtue training for truth-telling in AI models.

**Run ID:** `{run_id}`
**Approach:** {approach}
**Base model:** {base_model}

{description}

## Reproduction

{commands if commands else "See the run manifest for full reproduction commands."}

## Run Manifest

Full manifest: `runs/{run_id}/manifest.yaml` in the parrhesia repo.
"""
    return card


def generate_model_card(repo_id: str, run_id: str, metadata: dict) -> str:
    """Generate a HuggingFace model card with training metadata."""
    base_model = metadata.get("base_model", "Qwen/Qwen3-8B")
    approach = metadata.get("approach", "")
    description = metadata.get("description", "")
    hyperparameters = metadata.get("hyperparameters", {})
    training_data = metadata.get("training_data", {})
    eval_results = metadata.get("eval_results", {})

    # Build hyperparameters table
    hp_table = ""
    if hyperparameters:
        hp_table = "## Training Hyperparameters\n\n| Parameter | Value |\n|-----------|-------|\n"
        for k, v in hyperparameters.items():
            hp_table += f"| {k} | {v} |\n"

    # Build training data links
    data_section = ""
    if training_data:
        data_section = "## Training Data\n\n"
        for name, url in training_data.items():
            data_section += f"- **{name}**: [{url}]({url})\n"

    # Build eval results
    eval_section = ""
    if eval_results:
        eval_section = "## Evaluation Results\n\n| Dimension | Score |\n|-----------|-------|\n"
        for dim, score in eval_results.items():
            eval_section += f"| {dim} | {score} |\n"

    card = f"""\
---
license: apache-2.0
library_name: peft
base_model: {base_model}
tags:
  - parrhesia
  - sycophancy
  - character-training
  - lora
  - {run_id}
---

# {repo_id.split('/')[-1]}

LoRA adapter for reducing sycophancy in LLMs, part of the [parrhesia](https://github.com/daiostech/parrhesia) project.

**Run ID:** `{run_id}`
**Approach:** {approach}
**Base model:** `{base_model}`

{description}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}", revision="{run_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

{hp_table}
{data_section}
{eval_section}

## Run Manifest

Full manifest: `runs/{run_id}/manifest.yaml` in the parrhesia repo.
"""
    return card


def push_all(run_id: str, what: str = "all") -> dict[str, str]:
    """Push artifacts for a run. what can be 'all', 'data', 'adapters', or 'results'.

    Returns a dict of artifact name -> Hub URL.
    """
    from parrhesia.manifest import load_manifest

    manifest = load_manifest(run_id)
    urls = {}

    card_metadata = {
        "base_model": manifest.base_model,
        "approach": manifest.approach,
        "description": manifest.description,
        "steps": manifest.steps,
    }

    if what in ("all", "data"):
        # Push datasets based on what output files exist
        root = Path(__file__).resolve().parents[1]

        data_mappings = [
            ("data/generated/oct/dpo_pairs.jsonl", DATASET_REPOS["oct"], "dpo_pairs"),
            ("data/generated/oct/prompts.jsonl", DATASET_REPOS["oct"], "prompts"),
            ("data/generated/oct/chosen.jsonl", DATASET_REPOS["oct"], "chosen"),
            ("data/generated/oct/rejected.jsonl", DATASET_REPOS["oct"], "rejected"),
            ("data/generated/introspection/sft_introspection.jsonl", DATASET_REPOS["introspection"], "sft_introspection"),
            ("data/generated/introspection/self_reflections.jsonl", DATASET_REPOS["introspection"], "self_reflections"),
            ("data/generated/introspection/self_conversations.jsonl", DATASET_REPOS["introspection"], "self_conversations"),
            ("data/generated/introspection/principle_derivations.jsonl", DATASET_REPOS["introspection"], "principle_derivations"),
            ("data/generated/introspection/filtered.jsonl", DATASET_REPOS["introspection"], "filtered"),
        ]

        for rel_path, repo_id, split in data_mappings:
            full_path = root / rel_path
            if full_path.exists():
                url = push_dataset(full_path, repo_id, run_id, split, card_metadata)
                urls[f"data/{split}"] = url

    if what in ("all", "adapters"):
        root = Path(__file__).resolve().parents[1]
        adapters_dir = root / "adapters"

        # Collect hyperparameters from DPO/SFT steps
        hp = {}
        for step in manifest.steps:
            if step["step_name"] in ("dpo_training", "sft_introspection"):
                hp.update(step.get("hyperparameters", {}))

        model_card_meta = {**card_metadata, "hyperparameters": hp}

        # Scan adapters/ for any directory that looks like a LoRA adapter.
        # Repo id is derived from the directory name (daios/<name>), with
        # legacy lookups in MODEL_REPOS taking precedence for old paths.
        if adapters_dir.exists():
            for full_path in sorted(adapters_dir.iterdir()):
                if not full_path.is_dir():
                    continue
                if not (full_path / "adapter_config.json").exists():
                    continue
                repo_id = derive_repo_id(full_path)
                rel_path = full_path.relative_to(root)
                url = push_adapter(full_path, repo_id, run_id, model_card_meta)
                urls[f"adapter/{rel_path}"] = url

    if what in ("all", "results"):
        root = Path(__file__).resolve().parents[1]

        for results_file in (root / "results").glob("*.json"):
            model_name = results_file.stem
            url = push_eval_results(results_file, run_id, model_name)
            urls[f"results/{model_name}"] = url

    return urls


def pull_all(run_id: str, what: str = "all") -> dict[str, Path]:
    """Pull artifacts for a run. Returns dict of artifact name -> local path."""
    root = Path(__file__).resolve().parents[1]
    paths = {}

    if what in ("all", "data"):
        data_pulls = [
            (DATASET_REPOS["oct"], "data/generated/oct", "dpo_pairs"),
            (DATASET_REPOS["introspection"], "data/generated/introspection", "sft_introspection"),
        ]

        for repo_id, local_dir, split in data_pulls:
            try:
                local_path = pull_dataset(repo_id, root / local_dir / f"{split}.jsonl", run_id, split)
                paths[f"data/{split}"] = local_path
            except Exception:
                pass

    if what in ("all", "adapters"):
        # Read the run's manifest to learn which adapters this run produced
        # and where they live on Hub. Each entry maps a local-path label to
        # "<repo_id>@<branch>". We pull each one back to its original local
        # path (derived from the repo basename so adapter/<NAME>@<branch>
        # lands at adapters/<NAME>/), preserving the convention.
        try:
            from parrhesia.manifest import load_manifest

            manifest = load_manifest(run_id)
            hub_models = getattr(manifest, "hub_models", {}) or {}
        except Exception:
            hub_models = {}

        for label, ref in hub_models.items():
            # ref looks like "daios/parrhesia-sft-8b@run-007-B-qwen3-8b"
            if "@" in ref:
                repo_id, branch = ref.split("@", 1)
            else:
                repo_id, branch = ref, run_id

            # Local path is adapters/<repo basename>/
            repo_basename = repo_id.split("/")[-1]
            local_dir = root / "adapters" / repo_basename
            try:
                local_path = pull_adapter(repo_id, local_dir, branch)
                paths[f"adapter/{local_dir.relative_to(root)}"] = local_path
            except Exception as e:
                print(f"Warning: failed to pull {repo_id}@{branch}: {e}")

    if what in ("all", "results"):
        try:
            local_path = pull_dataset(EVAL_REPO, root / "results" / "eval.json", run_id, "results")
            paths["results"] = local_path
        except Exception:
            pass

    return paths
