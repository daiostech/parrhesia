#!/bin/bash
# RunPod evaluation sweep script for parrhesia run 4 (fire-and-forget)
#
# Runs 6-model weight sweep: baseline, DPO-only, and 4 merged weights (0.10-0.25).
# Each model gets quantitative + standard qualitative + hard qualitative.
#
# Prerequisites:
#   1. Create pod on RunPod (RTX 4090, 50GB container disk, PyTorch 2.x template)
#   2. SSH in, clone the repo, run setup:
#        cd /workspace && git clone https://github.com/daiostech/parrhesia.git && cd parrhesia
#        bash scripts/runpod_setup.sh
#   3. Upload .env:
#        mkdir -p /workspace/parrhesia
#        scp -P <PORT> path/to/your/.env root@<pod-ip>:/workspace/parrhesia/
#   4. Run: bash scripts/runpod_eval_sweep.sh
#
# Fire-and-forget mode (auto-stop pod when done):
#   AUTO_STOP=1 nohup bash scripts/runpod_eval_sweep.sh > eval.log 2>&1 &

set -euo pipefail

echo "=== Parrhesia Weight Sweep Evaluation (fire-and-forget) ==="

cd /workspace/parrhesia

# Activate serve venv
source .venv-serve/bin/activate

# Source env vars from .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
  echo "Loaded .env"
fi

RUN_ID="${PARRHESIA_RUN_ID:-run-004-A-qwen3-8b}"
echo "Run ID: $RUN_ID"

# --- Ensure adapters are available (pull from Hub if not on disk) ---
pull_if_missing() {
  local dir="$1" repo="$2"
  if [ -f "$dir/adapter_config.json" ]; then
    echo "  $dir already on disk, skipping pull."
  else
    echo "  Pulling $dir from Hub ($repo @ $RUN_ID)..."
    mkdir -p "$dir"
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$repo',
    repo_type='model',
    revision='$RUN_ID',
    local_dir='$dir',
    token=os.environ.get('HF_TOKEN'),
)
"
    echo "  Done."
  fi
}

echo ""
echo "Checking adapters..."

# DPO adapter
pull_if_missing "adapters/parrhesia-oct-8b" "daios/parrhesia-oct-8b"

# Merged adapters: repo has subfolders w000/, w010/, w015/, w020/, w025/
# Pull the whole repo once, then each subfolder is a complete adapter
SWEEP_DIR="adapters/parrhesia-merged-sweep"
if [ -f "$SWEEP_DIR/w000/adapter_config.json" ]; then
  echo "  $SWEEP_DIR already on disk, skipping pull."
else
  echo "  Pulling merged sweep adapters from Hub (daios/parrhesia-merged-8b @ $RUN_ID)..."
  mkdir -p "$SWEEP_DIR"
  python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='daios/parrhesia-merged-8b',
    repo_type='model',
    revision='$RUN_ID',
    local_dir='$SWEEP_DIR',
    token=os.environ.get('HF_TOKEN'),
)
"
  echo "  Done."
fi

# --- Start vLLM with base model + all adapters ---
# 6 models served from one instance:
#   "Qwen/Qwen3-8B"    = base model (no adapter)
#   "parrhesia-dpo"     = DPO-only adapter (= w000)
#   "parrhesia-w010"    = merged DPO*1.0 + SFT*0.10
#   "parrhesia-w015"    = merged DPO*1.0 + SFT*0.15
#   "parrhesia-w020"    = merged DPO*1.0 + SFT*0.20
#   "parrhesia-w025"    = merged DPO*1.0 + SFT*0.25
echo ""
echo "Starting vLLM server (base + 5 adapters)..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enable-lora \
  --lora-modules \
    parrhesia-dpo=./adapters/parrhesia-oct-8b \
    parrhesia-w010=./adapters/parrhesia-merged-sweep/w010 \
    parrhesia-w015=./adapters/parrhesia-merged-sweep/w015 \
    parrhesia-w020=./adapters/parrhesia-merged-sweep/w020 \
    parrhesia-w025=./adapters/parrhesia-merged-sweep/w025 \
  --max-lora-rank 64 \
  --port 8000 \
  --max-model-len 4096 &

VLLM_PID=$!

# Wait for vLLM to be ready (up to 8 min — base model + 5 LoRA adapters)
echo "Waiting for vLLM to start..."
for i in $(seq 1 240); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM is ready!"
    break
  fi
  if [ $i -eq 240 ]; then
    echo "ERROR: vLLM failed to start after 480s"
    kill $VLLM_PID 2>/dev/null
    exit 1
  fi
  sleep 2
done

# --- Run evaluation ---
RESULTS_DIR="results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# Define model→label mapping
# Format: "vllm_model_name:output_label"
MODELS=(
  "Qwen/Qwen3-8B:baseline"
  "parrhesia-dpo:dpo"
  "parrhesia-w010:merged-w010"
  "parrhesia-w015:merged-w015"
  "parrhesia-w020:merged-w020"
  "parrhesia-w025:merged-w025"
)

# Quantitative evaluation (6 models)
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

# Qualitative evaluation: standard + hard (6 models × 2 suites)
for entry in "${MODELS[@]}"; do
  MODEL_NAME="${entry%%:*}"
  LABEL="${entry##*:}"
  echo ""
  echo "=== Qualitative (standard): $MODEL_NAME ($LABEL) ==="
  OPENAI_API_BASE=http://localhost:8000/v1 \
    python -m parrhesia.cli qualitative \
    "$MODEL_NAME" \
    --output "$RESULTS_DIR/qualitative-${LABEL}.json"
  echo ""
  echo "=== Qualitative (hard): $MODEL_NAME ($LABEL) ==="
  OPENAI_API_BASE=http://localhost:8000/v1 \
    python -m parrhesia.cli qualitative \
    "$MODEL_NAME" \
    --golden-prompts parrhesia/benchmark/golden_prompts_hard.jsonl \
    --output "$RESULTS_DIR/qualitative-hard-${LABEL}.json"
done

# --- Stop vLLM ---
echo ""
echo "Stopping vLLM..."
kill $VLLM_PID 2>/dev/null

# --- Push results to Hub ---
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
echo "=== Weight sweep evaluation complete ==="
echo "Results in: $RESULTS_DIR/"
ls "$RESULTS_DIR/"*.json 2>/dev/null

# --- Self-stop the pod (opt-in via AUTO_STOP=1) ---
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
