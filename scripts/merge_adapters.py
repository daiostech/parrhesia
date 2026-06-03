#!/usr/bin/env python3
"""Weighted merge of DPO and SFT LoRA adapters.

Following OCT (Maiya et al. 2025): the final adapter is a weighted linear
combination of the DPO adapter (weight 1.0) and the SFT/introspection
adapter (weight 0.25).  Both adapters must target the same base model and
have the same rank/target modules.

Usage:
    python scripts/merge_adapters.py \
        --base-model Qwen/Qwen3-8B \
        --dpo-adapter adapters/parrhesia-oct-8b \
        --sft-adapter adapters/parrhesia-introspect-8b \
        --output adapters/parrhesia-merged-8b \
        --dpo-weight 1.0 \
        --sft-weight 0.25

The script loads the base model, attaches both adapters, creates a weighted
combination via PEFT's add_weighted_adapter, and saves the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapters(
    base_model: str,
    dpo_adapter: str,
    sft_adapter: str,
    output_dir: str,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
    run_id: str | None = None,
    push: bool = False,
):
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading DPO adapter: {dpo_adapter}")
    model = PeftModel.from_pretrained(model, dpo_adapter, adapter_name="dpo")

    print(f"Loading SFT adapter: {sft_adapter}")
    model.load_adapter(sft_adapter, adapter_name="sft")

    print(f"Merging: DPO * {dpo_weight} + SFT * {sft_weight}")
    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=[dpo_weight, sft_weight],
        adapter_name="persona",
        combination_type="linear",
    )

    # Set the merged adapter as active and save ONLY the merged "persona"
    # weights at the top level. Without selected_adapters, PEFT writes each
    # loaded adapter ("dpo", "sft", "persona") into its own subdirectory and
    # leaves no adapter_config.json at the top level — vLLM then can't find
    # the adapter when serving.
    model.set_adapter("persona")

    print(f"Saving merged adapter to: {output_dir}")
    model.save_pretrained(output_dir, selected_adapters=["persona"])
    tokenizer.save_pretrained(output_dir)

    # PEFT still nests selected adapters under their name (output_dir/persona/).
    # Flatten so adapter_config.json + adapter_model.safetensors live at the
    # top level — that's what vLLM and the rest of the pipeline expect.
    persona_dir = Path(output_dir) / "persona"
    if persona_dir.exists():
        import shutil
        for item in persona_dir.iterdir():
            target = Path(output_dir) / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        persona_dir.rmdir()

    print(f"Done. Merged adapter saved with weights: DPO={dpo_weight}, SFT={sft_weight}")

    # Push to Hub when a run is active (or --push given). Repo derived from
    # output dir basename so the local/Hub names stay aligned.
    import os
    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if (push or _run_id) and _run_id:
        from parrhesia.hub import push_adapter, derive_repo_id

        repo = derive_repo_id(output_dir)
        print(f"Pushing merged adapter to Hub: {repo}")
        url = push_adapter(output_dir, repo, _run_id)
        print(f"  → {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weighted merge of DPO and SFT LoRA adapters")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--dpo-adapter", required=True, help="Path to DPO LoRA adapter")
    parser.add_argument("--sft-adapter", required=True, help="Path to SFT/introspection LoRA adapter")
    parser.add_argument("--output", required=True, help="Output directory for merged adapter")
    parser.add_argument("--dpo-weight", type=float, default=1.0, help="Weight for DPO adapter (default: 1.0)")
    parser.add_argument("--sft-weight", type=float, default=0.25, help="Weight for SFT adapter (default: 0.25)")
    parser.add_argument("--run-id", default=None, help="Run ID for Hub branch (defaults to $PARRHESIA_RUN_ID)")
    parser.add_argument("--push", action="store_true", help="Push merged adapter to Hub (auto-on when run is active)")
    args = parser.parse_args()

    merge_adapters(
        base_model=args.base_model,
        dpo_adapter=args.dpo_adapter,
        sft_adapter=args.sft_adapter,
        output_dir=args.output,
        dpo_weight=args.dpo_weight,
        sft_weight=args.sft_weight,
        run_id=args.run_id,
        push=args.push,
    )
