#!/bin/bash
# RunPod introspection pipeline for parrhesia.
#
# Generates introspection data from a trained adapter (Run 7 default), then
# applies TWO filter variants in parallel — permissive (heuristic only) and
# strict (Claude judge) — producing two SFT data files for the dual-filter
# Run 8 experiment. Each variant is later trained as a separate LoRA and
# compared at the L0 probe.
#
# Usage:
#   bash scripts/runpod_introspection.sh [--adapter PATH] [--num-samples N] [--num-conversations N] [--min-score N]
#
#   --adapter             Path to LoRA adapter to introspect on
#                         (default: ./adapters/parrhesia-sft-8b — Run 7 SFT-v3)
#   --num-samples         Reflections per prompt (default: 100)
#   --num-conversations   Number of self-conversations (default: 200)
#   --min-score           Min judge total score 0-9 for strict filter (default: 5)
#
# Setup:
#   1. Pod: RTX 4090, 50GB container disk, PyTorch 2.x template
#   2. SSH in, clone the repo, run setup:
#        cd /workspace && git clone https://github.com/daiostech/parrhesia.git && cd parrhesia
#        bash scripts/runpod_setup.sh
#   3. Upload .env: scp -P <PORT> path/to/your/.env root@<pod-ip>:/workspace/parrhesia/
#   4. Pull adapter from Hub:
#        parrhesia hub-pull run-007-B-qwen3-8b --what adapters
#   5. Run: bash scripts/runpod_introspection.sh

set -euo pipefail

# --- Parse args ---
ADAPTER="./adapters/parrhesia-sft-8b"
NUM_SAMPLES=100
NUM_CONVERSATIONS=200
MIN_SCORE=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter)              ADAPTER="$2"; shift 2 ;;
    --num-samples)          NUM_SAMPLES="$2"; shift 2 ;;
    --num-conversations)    NUM_CONVERSATIONS="$2"; shift 2 ;;
    --min-score)            MIN_SCORE="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "=== Parrhesia Introspection Pipeline (dual filter) ==="
echo "Adapter:           $ADAPTER"
echo "Reflections:       ${NUM_SAMPLES} samples × 10 prompts"
echo "Conversations:     ${NUM_CONVERSATIONS} × 10 turns"
echo "Strict min_score:  $MIN_SCORE"
echo

cd /workspace/parrhesia

# Activate serve venv
source .venv-serve/bin/activate

# Redirect HuggingFace cache to /workspace (RunPod's root disk is typically
# only ~20 GB, while /workspace is the network volume with effectively
# unlimited space). Without this, downloading Qwen3-8B fills root and
# vLLM crashes with "No space left on device".
export HF_HOME="${HF_HOME:-/workspace/parrhesia/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Source PARRHESIA_RUN_ID from .env if set (enables auto-push to Hub)
if [ -f .env ] && grep -q PARRHESIA_RUN_ID .env; then
  export $(grep PARRHESIA_RUN_ID .env | xargs)
  echo "Run ID: $PARRHESIA_RUN_ID (auto-push enabled)"
fi

# Paths
RAW_DIR="data/generated/introspection"
PERMISSIVE_DIR="data/generated/introspection-permissive"
STRICT_DIR="data/generated/introspection-strict"

# --- Start vLLM with adapter ---
echo "Starting vLLM server with adapter $ADAPTER ..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enable-lora \
  --lora-modules parrhesia="$ADAPTER" \
  --max-lora-rank 64 \
  --port 8000 \
  --max-model-len 8192 &

VLLM_PID=$!

# Wait for vLLM (up to 10 min — model load + torch.compile + KV cache
# profiling + CUDA graph capture can take 3-5 min on cold start)
echo "Waiting for vLLM to start..."
for i in $(seq 1 300); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM is ready!"
    break
  fi
  if [ $i -eq 300 ]; then
    echo "ERROR: vLLM failed to start after 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
  fi
  sleep 2
done

# --- Stage 1: Generate raw introspection data (reflections + conversations) ---
echo ""
echo "=== Stage 1: Generation ==="

mkdir -p "$RAW_DIR"

echo "Generating self-reflections..."
python -m parrhesia.data.generate_introspection reflections \
  --model parrhesia \
  --api-base http://localhost:8000/v1 \
  --output-dir "$RAW_DIR" \
  --prompt-set oct \
  --num-samples "$NUM_SAMPLES" \
  --concurrency 16 \
  --no-thinking

echo ""
echo "Generating self-conversations..."
python -m parrhesia.data.generate_introspection conversations \
  --model parrhesia \
  --api-base http://localhost:8000/v1 \
  --output-dir "$RAW_DIR" \
  --interaction-style oct \
  --num-conversations "$NUM_CONVERSATIONS" \
  --num-turns 10 \
  --concurrency 8 \
  --no-thinking

# Stop vLLM — no longer needed for filter/format
echo ""
echo "Stopping vLLM (no longer needed for filter/format)..."
kill $VLLM_PID 2>/dev/null || true
sleep 5

# --- Stage 2: Stage raw files into per-variant directories ---
echo ""
echo "=== Stage 2: Stage raw files for each filter variant ==="

mkdir -p "$PERMISSIVE_DIR" "$STRICT_DIR"

# Symlink raw generation outputs into each variant's working dir.
# (Filter reads from the dir and writes filtered.jsonl + sft_introspection.jsonl
# back to the same dir.)
for f in self_reflections.jsonl self_conversations.jsonl principle_derivations.jsonl; do
  if [ -f "$RAW_DIR/$f" ]; then
    ln -sf "$(realpath "$RAW_DIR/$f")" "$PERMISSIVE_DIR/$f"
    ln -sf "$(realpath "$RAW_DIR/$f")" "$STRICT_DIR/$f"
  fi
done

# --- Stage 3: Filter variant A — permissive (heuristic only) ---
echo ""
echo "=== Stage 3a: Filter (permissive, heuristic only) ==="
python -m parrhesia.data.generate_introspection filter \
  --output-dir "$PERMISSIVE_DIR" \
  --filter-mode permissive

python -m parrhesia.data.generate_introspection format \
  --output-dir "$PERMISSIVE_DIR"

# --- Stage 3b: Filter variant B — strict (Claude judge) ---
echo ""
echo "=== Stage 3b: Filter (strict, Claude judge, min_score=$MIN_SCORE) ==="
python -m parrhesia.data.generate_introspection filter \
  --output-dir "$STRICT_DIR" \
  --filter-mode strict \
  --min-score "$MIN_SCORE"

python -m parrhesia.data.generate_introspection format \
  --output-dir "$STRICT_DIR"

# --- Summary ---
echo ""
echo "=== Introspection complete (dual filter) ==="
echo ""
echo "Permissive SFT data: $PERMISSIVE_DIR/sft_introspection.jsonl"
[ -f "$PERMISSIVE_DIR/sft_introspection.jsonl" ] && \
  echo "  examples: $(wc -l < "$PERMISSIVE_DIR/sft_introspection.jsonl")"
echo ""
echo "Strict SFT data:     $STRICT_DIR/sft_introspection.jsonl"
[ -f "$STRICT_DIR/sft_introspection.jsonl" ] && \
  echo "  examples: $(wc -l < "$STRICT_DIR/sft_introspection.jsonl")"
echo ""
echo "Next: bash scripts/runpod_train_run8.sh"
