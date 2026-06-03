"""
DPO training script for parrhesia character training using Unsloth.

This script trains a LoRA adapter on DPO preference pairs
(parrhesiastes chosen vs. base model rejected).

Usage:
    # Train on Qwen3-8B (development)
    python -m parrhesia.train.dpo \
        --model Qwen/Qwen3-8B \
        --data data/generated/oct/dpo_pairs.jsonl \
        --output adapters/parrhesia-oct-8b

    # Train on Qwen3-14B (production)
    python -m parrhesia.train.dpo \
        --model Qwen/Qwen3-14B \
        --data data/generated/oct/dpo_pairs.jsonl \
        --output adapters/parrhesia-oct-14b

Requires: pip install unsloth
Run on a machine with a GPU (RunPod RTX 4090 recommended).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be imported before trl
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

PatchDPOTrainer()

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer


def _patch_accelerate_device_check():
    """Monkey-patch accelerate's prepare_model to handle None device indices.

    Unsloth sets hf_device_map to {"": "cpu"} for 4-bit models.
    torch.device("cpu").index is None, which crashes accelerate's prepare_model.
    Clearing the map causes accelerate to call model.to(device), which OOMs.
    This patch: detect problematic values, temporarily clear the map AND disable
    device placement, then restore the map after prepare_model returns.
    """
    import accelerate

    _orig = accelerate.Accelerator.prepare_model

    def _safe_prepare_model(self, model, device_placement=None, evaluation_mode=False):
        saved_map = None
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            for v in model.hf_device_map.values():
                if isinstance(v, (str, torch.device)):
                    idx = torch.device(v).index if isinstance(v, str) else v.index
                else:
                    idx = v
                if idx is None:
                    saved_map = model.hf_device_map
                    model.hf_device_map = {}
                    device_placement = False
                    break
        result = _orig(self, model, device_placement=device_placement, evaluation_mode=evaluation_mode)
        if saved_map is not None:
            model.hf_device_map = saved_map
        return result

    accelerate.Accelerator.prepare_model = _safe_prepare_model


_patch_accelerate_device_check()


def load_dpo_dataset(data_path: str | Path) -> Dataset:
    """Load DPO pairs from JSONL into a HuggingFace Dataset."""
    data_path = Path(data_path)
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return Dataset.from_list(records)


def train(
    model_name: str = "Qwen/Qwen3-8B",
    data_path: str = "data/generated/oct/dpo_pairs.jsonl",
    output_dir: str = "adapters/parrhesia-oct",
    # LoRA config
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    # Training config
    num_epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    nll_loss_coef: float = 0.1,
    max_length: int = 1536,
    max_prompt_length: int = 512,
    max_seq_length: int = 2048,
    # Misc
    seed: int = 42,
    logging_steps: int = 1,
    save_steps: int = 100,
    warmup_ratio: float = 0.1,
    # Manifest
    run_id: str | None = None,
    push: bool = False,
    hub_repo: str | None = None,
):
    """Run DPO training with Unsloth."""

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=True,
        device_map={"": 0},
    )

    print("Configuring LoRA adapter")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
    )

    print(f"Loading dataset from: {data_path}")
    dataset = load_dpo_dataset(data_path)
    print(f"  {len(dataset)} training pairs loaded")

    # Build loss config: DPO + NLL auxiliary loss on chosen responses (like OCT)
    # NLL loss stabilizes training by keeping the model good at generating chosen responses
    dpo_config_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        warmup_ratio=warmup_ratio,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=seed,
        report_to="none",
    )

    if nll_loss_coef > 0:
        print(f"  DPO + NLL auxiliary loss (coef={nll_loss_coef})")
        dpo_config_kwargs["loss_type"] = ["sigmoid", "sft"]
        dpo_config_kwargs["loss_weights"] = [1.0, nll_loss_coef]

    training_args = DPOConfig(**dpo_config_kwargs)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles this (unloads adapter for ref)
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print("Starting DPO training...")
    train_result = trainer.train()

    print(f"Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Record step in manifest if run_id is active
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        hyperparameters = {
            "base_model": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "dpo_beta": beta,
            "nll_loss_coef": nll_loss_coef,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "max_seq_length": max_seq_length,
            "warmup_ratio": warmup_ratio,
            "optimizer": "adamw_8bit",
            "seed": seed,
        }

        metrics = {"training_pairs": len(dataset)}
        if hasattr(train_result, "metrics"):
            for k in ("train_loss", "train_runtime", "train_samples_per_second"):
                if k in train_result.metrics:
                    metrics[k] = train_result.metrics[k]

        command = _reconstruct_command(model_name, data_path, output_dir,
                                       lora_r, lora_alpha, num_epochs, batch_size,
                                       gradient_accumulation_steps, learning_rate,
                                       beta, nll_loss_coef, max_length,
                                       max_prompt_length, max_seq_length, seed)

        record_step(
            _run_id,
            step_name="dpo_training",
            command=command,
            hyperparameters=hyperparameters,
            inputs={"dpo_pairs": data_path},
            outputs={"adapter": output_dir},
            metrics=metrics,
        )
        print(f"Recorded step in manifest: {_run_id}")

    # Push adapter to Hub (auto-push when run is active, or explicit --push)
    if push or _run_id:
        from parrhesia.hub import push_adapter, MODEL_REPOS

        repo = hub_repo or MODEL_REPOS.get("oct-8b")
        if repo and _run_id:
            print(f"Pushing adapter to Hub: {repo}")
            url = push_adapter(output_dir, repo, _run_id)
            print(f"  → {url}")

    print("Done.")


def _reconstruct_command(model_name, data_path, output_dir, lora_r, lora_alpha,
                         num_epochs, batch_size, grad_accum, lr, beta,
                         nll_loss_coef, max_length, max_prompt_length,
                         max_seq_length, seed):
    """Reconstruct the full CLI command with all parameters (immune to default changes)."""
    return (
        f"python -m parrhesia.train.dpo"
        f" --model {model_name}"
        f" --data {data_path}"
        f" --output {output_dir}"
        f" --lora-r {lora_r}"
        f" --lora-alpha {lora_alpha}"
        f" --epochs {num_epochs}"
        f" --batch-size {batch_size}"
        f" --grad-accum {grad_accum}"
        f" --lr {lr}"
        f" --beta {beta}"
        f" --nll-loss-coef {nll_loss_coef}"
        f" --max-length {max_length}"
        f" --max-prompt-length {max_prompt_length}"
        f" --max-seq-length {max_seq_length}"
        f" --seed {seed}"
    )


def main():
    parser = argparse.ArgumentParser(description="DPO training for parrhesia")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/generated/oct/dpo_pairs.jsonl",
        help="Path to DPO pairs JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="adapters/parrhesia-oct",
        help="Output directory for adapter",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll-loss-coef", type=float, default=0.1,
                        help="NLL auxiliary loss coefficient on chosen responses (0 to disable)")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # Manifest + Hub
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for manifest tracking")
    parser.add_argument("--push", action="store_true", help="Push adapter to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default=None, help="Hub repo ID for adapter")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        nll_loss_coef=args.nll_loss_coef,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        run_id=args.run_id,
        push=args.push,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
