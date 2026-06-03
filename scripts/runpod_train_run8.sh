#!/bin/bash
# RunPod Run 8 training pipeline (dual filter variant).
#
# Trains TWO introspection LoRAs from the two filter variants produced by
# scripts/runpod_introspection.sh, then merges each with the Run 7 adapter
# to produce two shippable single-adapter checkpoints for the L0 probe.
#
# Architecture (per Run 3 pattern):
#   1. Load Qwen3-8B + Run 7 adapter (parrhesia-sft-8b)
#   2. Merge Run 7 into base weights (handled by parrhesia.train.sft when
#      --model points to an adapter directory — fresh LoRA is initialized)
#   3. Train fresh LoRA on introspection data (per filter variant)
#   4. Combine Run 7 LoRA + Run 8 LoRA via merge_adapters.py with weights 1.0+1.0
#      → single shippable adapter loaded onto vanilla Qwen3-8B at inference
#
# Usage:
#   bash scripts/runpod_train_run8.sh
#
# Setup:
#   - .venv-train (NOT .venv-serve) — training and serving have incompatible torch
#   - Run 7 adapter present at adapters/parrhesia-sft-8b
#   - SFT data at data/generated/introspection-{permissive,strict}/sft_introspection.jsonl
#     (produced by scripts/runpod_introspection.sh)

set -euo pipefail

cd /workspace/parrhesia

# Activate train venv
source .venv-train/bin/activate

# Redirect HuggingFace cache to /workspace so model weights don't fill the
# pod's small (~20 GB) root disk.
export HF_HOME="${HF_HOME:-/workspace/parrhesia/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Source PARRHESIA_RUN_ID from .env if set
if [ -f .env ] && grep -q PARRHESIA_RUN_ID .env; then
  export $(grep PARRHESIA_RUN_ID .env | xargs)
  echo "Run ID: $PARRHESIA_RUN_ID (auto-push enabled)"
fi

RUN7_ADAPTER="adapters/parrhesia-sft-8b"
PERMISSIVE_DATA="data/generated/introspection-permissive/sft_introspection.jsonl"
STRICT_DATA="data/generated/introspection-strict/sft_introspection.jsonl"

PERMISSIVE_LORA="adapters/parrhesia-introspect-permissive-8b"
STRICT_LORA="adapters/parrhesia-introspect-strict-8b"

PERMISSIVE_MERGED="adapters/parrhesia-introspect-permissive-merged-8b"
STRICT_MERGED="adapters/parrhesia-introspect-strict-merged-8b"

# Sanity checks
[ -d "$RUN7_ADAPTER" ] || { echo "ERROR: Run 7 adapter not found at $RUN7_ADAPTER"; exit 1; }
[ -f "$PERMISSIVE_DATA" ] || { echo "ERROR: Permissive SFT data not found at $PERMISSIVE_DATA"; exit 1; }
[ -f "$STRICT_DATA" ] || { echo "ERROR: Strict SFT data not found at $STRICT_DATA"; exit 1; }

echo "=== Run 8 Training (dual filter) ==="
echo "Run 7 base adapter:  $RUN7_ADAPTER"
echo "Permissive examples: $(wc -l < "$PERMISSIVE_DATA")"
echo "Strict examples:     $(wc -l < "$STRICT_DATA")"
echo

# Common SFT hyperparameters (Run 3 introspection recipe)
SFT_ARGS=(
  --model "$RUN7_ADAPTER"
  --lora-r 64
  --lora-alpha 128
  --epochs 1
  --batch-size 2
  --grad-accum 16
  --lr 5e-5
  --max-seq-length 3072
  --dtype bf16
  --no-4bit
)

# --- Stage 1a: Train Run 8a (permissive) LoRA ---
echo ""
echo "=== Stage 1a: SFT training (permissive variant) ==="
python -m parrhesia.train.sft \
  "${SFT_ARGS[@]}" \
  --data "$PERMISSIVE_DATA" \
  --output "$PERMISSIVE_LORA" \
  --run-id "${PARRHESIA_RUN_ID:-run-008-A-qwen3-8b}" \
  ${PUSH:+--push}

# --- Stage 1b: Train Run 8b (strict) LoRA ---
echo ""
echo "=== Stage 1b: SFT training (strict variant) ==="
python -m parrhesia.train.sft \
  "${SFT_ARGS[@]}" \
  --data "$STRICT_DATA" \
  --output "$STRICT_LORA" \
  --run-id "${PARRHESIA_RUN_ID:-run-008-A-qwen3-8b}" \
  ${PUSH:+--push}

# --- Stage 2: Merge each Run 8 LoRA with Run 7 LoRA (weights 1.0 + 1.0) ---
# We want the integrated character — full strength of habituation + full strength
# of self-knowledge — not a tunable mix.
echo ""
echo "=== Stage 2a: Merge Run 7 + Run 8a (permissive) ==="
python scripts/merge_adapters.py \
  --base-model Qwen/Qwen3-8B \
  --dpo-adapter "$RUN7_ADAPTER" \
  --sft-adapter "$PERMISSIVE_LORA" \
  --output "$PERMISSIVE_MERGED" \
  --dpo-weight 1.0 \
  --sft-weight 1.0

echo ""
echo "=== Stage 2b: Merge Run 7 + Run 8b (strict) ==="
python scripts/merge_adapters.py \
  --base-model Qwen/Qwen3-8B \
  --dpo-adapter "$RUN7_ADAPTER" \
  --sft-adapter "$STRICT_LORA" \
  --output "$STRICT_MERGED" \
  --dpo-weight 1.0 \
  --sft-weight 1.0

# --- Summary ---
echo ""
echo "=== Run 8 training complete ==="
echo ""
echo "Output adapters (loaded onto vanilla Qwen3-8B at inference):"
echo "  $PERMISSIVE_MERGED  (Run 8a — heuristic-only filter)"
echo "  $STRICT_MERGED      (Run 8b — Claude judge filter)"
echo ""
echo "Run 7 LoRA + Run 8 LoRA fresh adapters (intermediate, optional to keep):"
echo "  $PERMISSIVE_LORA"
echo "  $STRICT_LORA"
echo ""
echo "Next: bash scripts/runpod_eval.sh with both merged adapters as the comparison set."
