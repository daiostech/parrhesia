#!/bin/bash
# General-purpose RunPod evaluation script for parrhesia (fire-and-forget).
#
# Evaluates a base model + any number of LoRA adapters. Runs quantitative +
# standard qualitative + hard qualitative for every model.
#
# Usage:
#   # Approach B: base + 1 adapter (SFT training data for contamination check)
#   bash scripts/runpod_eval.sh \
#     --run-id run-006-B-qwen3-8b \
#     --base-model Qwen/Qwen3-8B \
#     --adapters "parrhesia-sft=./adapters/parrhesia-sft-8b" \
#     --training-data data/generated/sft/training_pairs_revised.jsonl \
#     --prompt-key messages.0.content
#
#   # Approach A sweep: base + 5 adapters (OCT training data — the default)
#   bash scripts/runpod_eval.sh \
#     --run-id run-004-A-qwen3-8b \
#     --base-model Qwen/Qwen3-8B \
#     --adapters "parrhesia-dpo=./adapters/parrhesia-oct-8b parrhesia-w010=./adapters/parrhesia-merged-sweep/w010 ..."
#
# Contamination policy:
#   - Baseline iteration: skips the contamination check by default (untrained
#     model — there is no training data to be contaminated by). Pass
#     --legacy-baseline-contamination to force the check on baseline; this is
#     used by Run 6/7 manifest backfills to reproduce pre-fix behavior.
#   - Adapter iterations: always run the contamination check.
#
# Fire-and-forget mode (auto-stop pod when done):
#   AUTO_STOP=1 nohup bash scripts/runpod_eval.sh \
#     --run-id run-005-B-qwen3-8b \
#     --base-model Qwen/Qwen3-8B \
#     --adapters "parrhesia-sft=./adapters/parrhesia-sft-8b" \
#     > eval.log 2>&1 &
#
# Prerequisites:
#   1. Create pod on RunPod (RTX 4090, 50GB container disk, PyTorch 2.x template)
#   2. SSH in, clone the repo, run setup:
#        cd /workspace && git clone https://github.com/daiostech/parrhesia.git && cd parrhesia
#        bash scripts/runpod_setup.sh
#   3. Upload .env:
#        mkdir -p /workspace/parrhesia
#        scp -P <PORT> path/to/your/.env root@<pod-ip>:/workspace/parrhesia/
#   4. Activate serve venv: source .venv-serve/bin/activate
#   5. Run this script

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

RUN_ID=""
BASE_MODEL=""
ADAPTERS_STR=""
TRAINING_DATA=""
PROMPT_KEY=""
LEGACY_BASELINE_CONTAMINATION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --adapters)
      ADAPTERS_STR="$2"
      shift 2
      ;;
    --training-data)
      TRAINING_DATA="$2"
      shift 2
      ;;
    --prompt-key)
      PROMPT_KEY="$2"
      shift 2
      ;;
    --legacy-baseline-contamination)
      # Reproduces pre-fix behavior: contamination check applied to baseline
      # iteration as well. Backfilled into Run 6/7 manifests so replays match
      # the original (buggy) skip-pattern.
      LEGACY_BASELINE_CONTAMINATION=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/runpod_eval.sh --run-id <id> --base-model <model> --adapters \"name=path ...\" [--training-data <file>] [--prompt-key <key>] [--legacy-baseline-contamination]"
      exit 1
      ;;
  esac
done

if [ -z "$RUN_ID" ] || [ -z "$BASE_MODEL" ]; then
  echo "ERROR: --run-id and --base-model are required."
  echo "Usage: bash scripts/runpod_eval.sh --run-id <id> --base-model <model> [--adapters \"name=path ...\"] [--training-data <file>] [--prompt-key <key>] [--legacy-baseline-contamination]"
  exit 1
fi

echo "=== Parrhesia Evaluation ==="
echo "Run ID:         $RUN_ID"
echo "Base model:     $BASE_MODEL"
echo "Adapters:       ${ADAPTERS_STR:-none}"
echo "Training data:  ${TRAINING_DATA:-default (data/generated/oct/prompts.jsonl)}"
echo "Prompt key:     ${PROMPT_KEY:-default (prompt)}"
echo "Legacy baseline contamination: $LEGACY_BASELINE_CONTAMINATION"

cd /workspace/parrhesia

# Source env vars from .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "Loaded .env"
fi

# Redirect HuggingFace cache to /workspace so model weights don't fill the
# pod's small (~20 GB) root disk.
export HF_HOME="${HF_HOME:-/workspace/parrhesia/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Override run ID from CLI (takes precedence over .env)
export PARRHESIA_RUN_ID="$RUN_ID"

# ---------------------------------------------------------------------------
# Ensure the serve venv is active
# ---------------------------------------------------------------------------
# This script invokes bare `python` (vLLM, the benchmark client, the judge,
# the CLI) and expects .venv-serve. If it was launched without activating it,
# fall back to sourcing it so the run doesn't fail cryptically with
# "No module named 'vllm'". Error clearly if setup was never run.
if ! python -m pip show vllm >/dev/null 2>&1; then
  if [ -f .venv-serve/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv-serve/bin/activate
    echo "Auto-activated serve venv: $(command -v python)"
  else
    echo "ERROR: vLLM not available and .venv-serve missing. Run scripts/runpod_setup.sh first, or 'source .venv-serve/bin/activate'." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Build vLLM launch command
# ---------------------------------------------------------------------------

# Parse adapters into arrays
ADAPTER_NAMES=()
ADAPTER_PATHS=()
LORA_MODULES_ARGS=()

if [ -n "$ADAPTERS_STR" ]; then
  for entry in $ADAPTERS_STR; do
    name="${entry%%=*}"
    path="${entry##*=}"
    ADAPTER_NAMES+=("$name")
    ADAPTER_PATHS+=("$path")
    LORA_MODULES_ARGS+=("$name=$path")
  done
fi

# Build vLLM command
VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$BASE_MODEL"
  --port 8000
  --max-model-len 8192
)

if [ ${#LORA_MODULES_ARGS[@]} -gt 0 ]; then
  VLLM_CMD+=(--enable-lora --max-lora-rank 64 --lora-modules "${LORA_MODULES_ARGS[@]}")
fi

echo ""
echo "Starting vLLM server..."
"${VLLM_CMD[@]}" &
VLLM_PID=$!

# Wait for vLLM to be ready (up to 10 min — multi-adapter loading +
# torch.compile + CUDA graph capture can be slow on cold start)
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

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

RESULTS_DIR="results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# Build model list: baseline + each adapter
# Format: "vllm_model_name:output_label"
MODELS=("${BASE_MODEL}:baseline")
for name in "${ADAPTER_NAMES[@]}"; do
  MODELS+=("${name}:${name}")
done

echo ""
echo "Models to evaluate: ${#MODELS[@]}"
for entry in "${MODELS[@]}"; do
  echo "  - ${entry}"
done

# Quantitative evaluation
for entry in "${MODELS[@]}"; do
  MODEL_NAME="${entry%%:*}"
  LABEL="${entry##*:}"
  echo ""
  echo "=== Quantitative: $MODEL_NAME ($LABEL) ==="
  OPENAI_API_BASE=http://localhost:8000/v1 \
    python -m parrhesia.benchmark.evaluate \
    "$MODEL_NAME" \
    --output "$RESULTS_DIR/$LABEL.json"
done

# Qualitative evaluation: standard + hard
for entry in "${MODELS[@]}"; do
  MODEL_NAME="${entry%%:*}"
  LABEL="${entry##*:}"
  echo ""
  QUAL_ARGS=()
  if [ -n "$TRAINING_DATA" ]; then
    QUAL_ARGS+=(--training-data "$TRAINING_DATA")
  fi
  if [ -n "$PROMPT_KEY" ]; then
    QUAL_ARGS+=(--prompt-key "$PROMPT_KEY")
  fi

  # Per-model contamination policy:
  # - Baseline (untrained) defaults to skip; --legacy-baseline-contamination
  #   forces the check on baseline (reproduces pre-fix Run 6/7 behavior).
  # - Adapters always check, regardless of legacy flag (their training data
  #   may topically overlap golden prompts).
  if [ "$LABEL" = "baseline" ]; then
    if [ "$LEGACY_BASELINE_CONTAMINATION" = "true" ]; then
      QUAL_ARGS+=(--no-skip-contamination)
    else
      QUAL_ARGS+=(--skip-contamination)
    fi
  else
    QUAL_ARGS+=(--no-skip-contamination)
  fi

  echo "=== Qualitative (standard): $MODEL_NAME ($LABEL) ==="
  OPENAI_API_BASE=http://localhost:8000/v1 \
    python -m parrhesia.cli qualitative \
    "$MODEL_NAME" \
    --output "$RESULTS_DIR/qualitative-${LABEL}.json" \
    "${QUAL_ARGS[@]}"
  echo ""
  echo "=== Qualitative (hard): $MODEL_NAME ($LABEL) ==="
  OPENAI_API_BASE=http://localhost:8000/v1 \
    python -m parrhesia.cli qualitative \
    "$MODEL_NAME" \
    --golden-prompts parrhesia/benchmark/golden_prompts_hard.jsonl \
    --output "$RESULTS_DIR/qualitative-hard-${LABEL}.json" \
    "${QUAL_ARGS[@]}"
done

# ---------------------------------------------------------------------------
# Stop vLLM
# ---------------------------------------------------------------------------

echo ""
echo "Stopping vLLM..."
kill $VLLM_PID 2>/dev/null

# ---------------------------------------------------------------------------
# Push results to Hub
# ---------------------------------------------------------------------------

echo ""
echo "Pushing results to Hub..."
python -c "
from huggingface_hub import HfApi
import os, glob

api = HfApi(token=os.environ.get('HF_TOKEN'))
repo_id = 'daios/parrhesia-eval-results'
run_id = '$RUN_ID'

api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
try:
    api.create_branch(repo_id=repo_id, branch=run_id, repo_type='dataset')
except:
    pass

for f in sorted(glob.glob('$RESULTS_DIR/*.json')):
    fname = os.path.basename(f)
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f'results/{fname}',
        repo_id=repo_id,
        repo_type='dataset',
        revision=run_id,
        commit_message=f'Upload {fname} from {run_id}',
    )
    print(f'  Pushed: {fname}')

print('Results pushed to Hub.')
"

echo ""
echo "=== Evaluation complete ==="
echo "Results in: $RESULTS_DIR/"
ls "$RESULTS_DIR/"*.json 2>/dev/null

# ---------------------------------------------------------------------------
# Self-stop the pod (opt-in via AUTO_STOP=1)
# ---------------------------------------------------------------------------

if [ "${AUTO_STOP:-0}" = "1" ]; then
  echo ""
  echo "Stopping pod..."
  if command -v runpodctl &> /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
    echo "Pod stop requested."
  else
    echo "WARNING: Could not auto-stop pod (runpodctl not found or RUNPOD_POD_ID not set)."
    echo "Kill the pod manually."
  fi
fi
