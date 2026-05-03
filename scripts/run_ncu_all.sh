#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found in PATH."
  echo "Please install Nsight Compute and ensure 'ncu --version' works."
  exit 1
fi

run_one_ncu() {
  local proj="$1"
  echo "========== [$proj] ncu profile =========="
  pushd "$ROOT_DIR/$proj" >/dev/null
  bash project-proof/scripts/profile_ncu.sh
  echo "========== [$proj] ncu plots =========="
  python project-proof/scripts/plot_ncu_summary.py
  popd >/dev/null
}

run_one_ncu "softmax"
run_one_ncu "gemv"
run_one_ncu "int8-quantize"

echo "All NCU profiling + plotting jobs completed."
