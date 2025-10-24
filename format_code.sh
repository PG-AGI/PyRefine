#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -x "${SCRIPT_DIR}/env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/env/Scripts/python.exe"
elif [[ -x "${SCRIPT_DIR}/env/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/env/bin/python"
fi

if [[ $# -eq 0 ]]; then
    set -- --all
fi

echo "Running Python formatting toolchain via ${PYTHON_BIN}..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/formatting.py" "$@"
echo "Formatting and linting completed successfully."
