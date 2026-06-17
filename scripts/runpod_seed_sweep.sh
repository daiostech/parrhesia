#!/bin/bash
# Multi-seed training sweep for parrhesia confidence intervals.
#
# Retrains the Run 7 SFT recipe across several seeds (the v010 arm), and
# optionally the v0.2.0-constitution recipe (the v020 arm), to measure
# TRAINING-seed variance and put a confidence interval on the headline delta.
# Each seed varies LoRA init + data shuffling (parrhesia.train.sft --seed),
# holding the training data and every hyperparameter fixed. Seed 42 reproduces
# the shipped Run 7 adapter exactly (it was trained at the default seed 42),
# so it doubles as a reproduction anchor.
#
# Reproducibility: every training run and every eval run gets a real manifest
# at runs/<run-id>/manifest.yaml (created here so sft.py/evaluate.py record into
# it), exactly like runs 001-009. All hyperparameters are flags (defaults pin
# the Run 7 recipe), not hardcoded literals.
#
# This is the TRAIN phase (.venv-train). It trains each seed's adapter to
# adapters/<name>-s<seed> and then WRITES the matching eval command(s) to
# seed_sweep_eval.sh for the EVAL phase (.venv-serve) — training (unsloth) and
# serving (vLLM) need incompatible torch and cannot share a venv.
#
# Usage:
#   source .venv-train/bin/activate
#   bash scripts/runpod_seed_sweep.sh --arms both --seeds "42 1 2 3 4"
#   # then, EVAL phase (different venv):
#   source .venv-serve/bin/activate && bash seed_sweep_eval.sh
#   # then, ANALYSIS (local or pod):
#   python -m parrhesia.benchmark.stats --aggregate results/<prefix>-v010-eval \
#          [--equivalence results/<prefix>-v020-eval]

set -euo pipefail

# --- Defaults (the Run 7 recipe; override any via flags) ---
ARMS="both"
SEEDS="42 1 2 3 4"
BASE_MODEL="Qwen/Qwen3-8B"
RUN_PREFIX="run-010-B-qwen3-8b"
PUSH_FLAG=""
LORA_R=64
LORA_ALPHA=128
EPOCHS=3
BATCH=2
GRAD_ACCUM=16
LR=2e-4
MAXSEQ=2048
V010_DATA="data/generated/sft/training_pairs_revised.jsonl"
V010_NAME="parrhesia-sft"
V020_DATA="data/generated/sft-v020/training_pairs_revised.jsonl"
V020_NAME="parrhesia-sft-v020"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --run-prefix) RUN_PREFIX="$2"; shift 2;;
    --push) PUSH_FLAG="--push"; shift;;          # archive each seed adapter to the Hub (optional)
    --lora-r) LORA_R="$2"; shift 2;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --max-seq-length) MAXSEQ="$2"; shift 2;;
    --v010-data) V010_DATA="$2"; shift 2;;
    --v020-data) V020_DATA="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

case "$ARMS" in
  both) ARM_LIST="v010 v020";;
  v010) ARM_LIST="v010";;
  v020) ARM_LIST="v020";;
  *) echo "ERROR: --arms must be one of: v010 | v020 | both"; exit 1;;
esac

arm_data() { case "$1" in v010) echo "$V010_DATA";; v020) echo "$V020_DATA";; esac; }
arm_name() { case "$1" in v010) echo "$V010_NAME";; v020) echo "$V020_NAME";; esac; }

# Create runs/<run-id>/manifest.yaml if absent, so sft.py / evaluate.py
# record_step() populate it (they no-op when the manifest is missing).
create_manifest_for() {  # run_id  base_model  description
  python - "$1" "$2" "$3" <<'PY' || true
import sys
from parrhesia.manifest import create_manifest, _manifest_path
rid, base, desc = sys.argv[1], sys.argv[2], sys.argv[3]
if _manifest_path(rid).exists():
    print(f"[manifest] runs/{rid}/manifest.yaml exists")
else:
    create_manifest(rid, "B", base, desc)
    print(f"[manifest] created runs/{rid}/manifest.yaml")
PY
}

cd /workspace/parrhesia

if [ -f .env ]; then set -a; source .env; set +a; echo "Loaded .env"; fi
export HF_HOME="${HF_HOME:-/workspace/parrhesia/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Ensure the TRAIN venv (unsloth) is active, not the serve venv.
if ! python -m pip show unsloth >/dev/null 2>&1; then
  if [ -f .venv-train/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv-train/bin/activate
    echo "Auto-activated train venv: $(command -v python)"
  else
    echo "ERROR: unsloth not available and .venv-train missing. Run scripts/runpod_setup.sh, then 'source .venv-train/bin/activate'." >&2
    exit 1
  fi
fi

echo "=== Parrhesia multi-seed sweep — TRAIN phase ==="
echo "Arms:    $ARM_LIST"
echo "Seeds:   $SEEDS"
echo "Base:    $BASE_MODEL"
echo "Prefix:  $RUN_PREFIX"
echo "Recipe:  lora_r=$LORA_R alpha=$LORA_ALPHA epochs=$EPOCHS batch=$BATCH grad_accum=$GRAD_ACCUM lr=$LR max_seq=$MAXSEQ"
echo "Push:    ${PUSH_FLAG:-(local only)}"

# Validate training data is present before spending any GPU time.
for arm in $ARM_LIST; do
  data="$(arm_data "$arm")"
  [ -f "$data" ] || { echo "ERROR: $arm training data not found at '$data' — scp it to the pod first."; exit 1; }
done

# --- Train every (arm, seed) ---
for arm in $ARM_LIST; do
  data="$(arm_data "$arm")"
  name="$(arm_name "$arm")"
  for seed in $SEEDS; do
    out="adapters/${name}-s${seed}"
    runid="${RUN_PREFIX}-${arm}-s${seed}"
    echo ""
    echo "=== TRAIN arm=$arm seed=$seed -> $out (run-id $runid) ==="
    create_manifest_for "$runid" "$BASE_MODEL" "Multi-seed CI sweep | arm $arm | seed $seed | data $data"
    python -m parrhesia.train.sft \
      --model "$BASE_MODEL" \
      --data "$data" \
      --output "$out" \
      --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM" \
      --lr "$LR" --max-seq-length "$MAXSEQ" \
      --seed "$seed" \
      --run-id "$runid" \
      $PUSH_FLAG
  done
  # Pre-create the arm's eval manifest so evaluate.py records into it.
  evalid="${RUN_PREFIX}-${arm}-eval"
  create_manifest_for "$evalid" "$BASE_MODEL" "Multi-seed CI sweep eval | arm $arm | seeds: $SEEDS"
done

# --- Emit the EVAL-phase script (runs in .venv-serve) ---
EVAL_SCRIPT="seed_sweep_eval.sh"
{
  echo "#!/bin/bash"
  echo "# Auto-generated by runpod_seed_sweep.sh. Run in .venv-serve."
  echo "# Evaluates baseline + all seed adapters per arm; records into runs/<id>/manifest.yaml; pushes results to the Hub."
  echo "set -euo pipefail"
  echo "source .venv-serve/bin/activate"
} > "$EVAL_SCRIPT"

for arm in $ARM_LIST; do
  data="$(arm_data "$arm")"
  name="$(arm_name "$arm")"
  adapters=""
  for seed in $SEEDS; do
    adapters="${adapters}${name}-s${seed}=./adapters/${name}-s${seed} "
  done
  evalid="${RUN_PREFIX}-${arm}-eval"
  echo "bash scripts/runpod_eval.sh --run-id ${evalid} --base-model ${BASE_MODEL} --adapters \"${adapters}\" --training-data ${data} --prompt-key messages.0.content" >> "$EVAL_SCRIPT"
done
chmod +x "$EVAL_SCRIPT"

echo ""
echo "=== TRAIN phase complete ==="
echo "Manifests created under runs/${RUN_PREFIX}-* (sft steps recorded; eval steps recorded by the eval phase)."
echo "Wrote $EVAL_SCRIPT — switch venvs and run it:"
echo "    source .venv-serve/bin/activate && bash $EVAL_SCRIPT"
echo ""
echo "Then analyze (local or pod):"
for arm in $ARM_LIST; do
  echo "    python -m parrhesia.benchmark.stats --aggregate results/${RUN_PREFIX}-${arm}-eval"
done
if [ "$ARMS" = "both" ]; then
  echo "    python -m parrhesia.benchmark.stats --aggregate results/${RUN_PREFIX}-v010-eval --equivalence results/${RUN_PREFIX}-v020-eval"
fi
