#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if ! "$VENV_PYTHON" -c "import snaptui" 2>/dev/null; then
    echo "Installing snaptui..."
    "$SCRIPT_DIR/.venv/bin/pip" install "snaptui>=0.1.2"
fi
exec "$VENV_PYTHON" -m data_viewer "$@"
