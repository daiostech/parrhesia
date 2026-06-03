"""
SFT training script for parrhesia character training using Unsloth.

Used for:
- Approach B (Direct SFT): Train directly on curated virtue/vice conversation pairs
- Approach C Phase 1 (SFT+DPO): SFT first, then DPO refinement on top

Usage:
    python -m parrhesia.train.sft \
        --model Qwen/Qwen3-8B \
        --data data/generated/sft/training_pairs.jsonl \
        --output adapters/parrhesia-sft-8b

Requires: pip install unsloth
Run on a machine with a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import Dataset
from trl import SFTConfig, SFTTrainer


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


def load_sft_dataset(data_path: str | Path) -> Dataset:
    """
    Load SFT training data from JSONL.

    Expected format (conversational):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Or multi-turn:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """
    data_path = Path(data_path)
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return Dataset.from_list(records)


def train(
    model_name: str = "Qwen/Qwen3-8B",
    data_path: str = "data/generated/sft/training_pairs.jsonl",
    output_dir: str = "adapters/parrhesia-sft",
    # LoRA config
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    # Training config
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    # Misc
    seed: int = 42,
    logging_steps: int = 1,
    save_steps: int = 100,
    warmup_ratio: float = 0.05,
    # Adapter handling
    continue_adapter: bool = False,
    # Model loading
    load_in_4bit: bool = True,
    dtype: str | None = None,  # "bf16", "fp16", or None (auto-detect)
    # Manifest
    run_id: str | None = None,
    push: bool = False,
    hub_repo: str | None = None,
):
    """Run SFT training with Unsloth."""
    import torch

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map.get(dtype) if dtype else None

    print(f"Loading model: {model_name}")
    print(f"  load_in_4bit={load_in_4bit}, dtype={dtype or 'auto'}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        device_map={"": 0},
    )

    # Handle existing LoRA adapters (e.g. loading a DPO adapter for introspection SFT)
    has_existing_adapter = hasattr(model, "peft_config")

    if has_existing_adapter and continue_adapter:
        print("Existing LoRA adapter detected — continuing training on it (--continue-adapter)")
        print("WARNING: This modifies the existing adapter weights directly.")
        FastLanguageModel.for_training(model)
    else:
        if has_existing_adapter:
            print("Existing LoRA adapter detected — merging into base weights")
            print("  (DPO signal preserved in base; fresh LoRA for SFT)")
            model = model.merge_and_unload()

        # Set default dtype for LoRA weight initialization to match training
        # dtype — prevents mismatch in unsloth's fast_lora backward pass
        train_dtype = torch_dtype or (torch.bfloat16 if is_bfloat16_supported() else torch.float16)
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(train_dtype)

        print(f"Configuring new LoRA adapter (r={lora_r}, alpha={lora_alpha}, dtype={train_dtype})")
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

        torch.set_default_dtype(_prev_dtype)

    print(f"Loading dataset from: {data_path}")
    dataset = load_sft_dataset(data_path)
    print(f"  {len(dataset)} training examples loaded")

    def formatting_func(examples):
        """Format messages using the tokenizer's chat template."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        warmup_ratio=warmup_ratio,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=seed,
        dataset_text_field="text",
        report_to="none",
    )

    dataset = dataset.map(formatting_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    print("Starting SFT training...")
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
            "max_seq_length": max_seq_length,
            "warmup_ratio": warmup_ratio,
            "optimizer": "adamw_8bit",
            "seed": seed,
        }

        metrics = {"training_examples": len(dataset)}
        if hasattr(train_result, "metrics"):
            for k in ("train_loss", "train_runtime", "train_samples_per_second"):
                if k in train_result.metrics:
                    metrics[k] = train_result.metrics[k]

        command = _reconstruct_command(model_name, data_path, output_dir,
                                       lora_r, lora_alpha, num_epochs, batch_size,
                                       gradient_accumulation_steps, learning_rate,
                                       max_seq_length, seed)

        record_step(
            _run_id,
            step_name="sft_training",
            command=command,
            hyperparameters=hyperparameters,
            inputs={"sft_data": data_path},
            outputs={"adapter": output_dir},
            metrics=metrics,
        )
        print(f"Recorded step in manifest: {_run_id}")

    # Push adapter to Hub (auto-push when run is active, or explicit --push)
    if push or _run_id:
        from parrhesia.hub import push_adapter, derive_repo_id

        # Derive repo from output directory basename (adapters/X → daios/X);
        # --hub-repo overrides for cases that need a specific destination.
        repo = hub_repo or derive_repo_id(output_dir)
        if repo and _run_id:
            print(f"Pushing adapter to Hub: {repo}")
            url = push_adapter(output_dir, repo, _run_id)
            print(f"  → {url}")

    print("Done.")


def _reconstruct_command(model_name, data_path, output_dir, lora_r, lora_alpha,
                         num_epochs, batch_size, grad_accum, lr,
                         max_seq_length, seed):
    """Reconstruct the full CLI command with all parameters."""
    return (
        f"python -m parrhesia.train.sft"
        f" --model {model_name}"
        f" --data {data_path}"
        f" --output {output_dir}"
        f" --lora-r {lora_r}"
        f" --lora-alpha {lora_alpha}"
        f" --epochs {num_epochs}"
        f" --batch-size {batch_size}"
        f" --grad-accum {grad_accum}"
        f" --lr {lr}"
        f" --max-seq-length {max_seq_length}"
        f" --seed {seed}"
    )


def main():
    parser = argparse.ArgumentParser(description="SFT training for parrhesia")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/generated/sft/training_pairs.jsonl",
        help="Path to SFT training JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="adapters/parrhesia-sft",
        help="Output directory for adapter",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # Adapter handling
    parser.add_argument("--continue-adapter", action="store_true",
                        help="Continue training existing LoRA adapter instead of merging and creating fresh one")

    # Model loading
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (use full precision)")
    parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16"],
                        help="Model dtype: bf16, fp16, or auto-detect (default)")

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
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        continue_adapter=args.continue_adapter,
        load_in_4bit=not args.no_4bit,
        dtype=args.dtype,
        run_id=args.run_id,
        push=args.push,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
