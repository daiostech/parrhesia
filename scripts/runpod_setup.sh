#!/bin/bash
# RunPod setup for parrhesia
# Creates two virtual environments (train + serve) with isolated dependencies.
#
# Usage:
#   1. Create pod on RunPod (RTX 4090, 50GB container disk, PyTorch 2.x template)
#   2. SSH in: ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
#   3. Clone repo: cd /workspace && git clone https://github.com/daiostech/parrhesia.git
#   4. Run: cd parrhesia && bash scripts/runpod_setup.sh
#   5. Upload .env and data (scp from local machine)

set -euo pipefail

echo "=== Parrhesia Setup ==="

cd /workspace/parrhesia

# Show current system PyTorch/CUDA versions
echo ""
python -c "import torch; print(f'System PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Create data directories
mkdir -p data/generated/oct data/generated/introspection

# --- Train venv (unsloth + trl + peft) ---
echo "Creating train venv..."
python -m venv .venv-train
.venv-train/bin/pip install --upgrade pip -q

echo "Installing training dependencies..."
.venv-train/bin/pip install -r requirements-train.txt -q

echo "Syncing unsloth..."
.venv-train/bin/pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo

echo "Installing parrhesia package..."
.venv-train/bin/pip install -e . -q

echo ""
.venv-train/bin/python -c "import torch; print(f'Train venv — PyTorch: {torch.__version__}')"
.venv-train/bin/python -c "import unsloth; print(f'Train venv — Unsloth: {unsloth.__version__}')"

# --- Serve venv (vLLM + openai + anthropic) ---
echo ""
echo "Creating serve venv..."
python -m venv .venv-serve
.venv-serve/bin/pip install --upgrade pip -q

echo "Installing serving dependencies..."
.venv-serve/bin/pip install -r requirements-serve.txt -q

echo "Installing parrhesia package..."
.venv-serve/bin/pip install -e . -q

echo ""
.venv-serve/bin/python -c "import torch; print(f'Serve venv — PyTorch: {torch.__version__}')"
.venv-serve/bin/python -c "import vllm; print(f'Serve venv — vLLM: {vllm.__version__}')"

# Source HF_TOKEN from .env if available
if [ -f .env ]; then
    export $(grep HF_TOKEN .env 2>/dev/null | xargs)
    if [ -n "${HF_TOKEN:-}" ]; then
        echo ""
        echo "HF_TOKEN loaded from .env"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Two virtual environments created:"
echo "  .venv-train  — for DPO and SFT training (unsloth)"
echo "  .venv-serve  — for introspection and eval (vLLM)"
echo ""
echo "Activate with:"
echo "  source .venv-train/bin/activate   # before training"
echo "  source .venv-serve/bin/activate   # before serving/eval"
echo ""
echo "Or run scripts directly (they activate the correct venv)."
