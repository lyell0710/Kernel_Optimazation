#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_PATH="$ROOT_DIR/build/int8_quantize_bench"
OUT_DIR="$ROOT_DIR/project-proof/profiling/ncu"
OUT_BASE="$OUT_DIR/int8_quantize_ncu"

mkdir -p "$OUT_DIR"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. Please install Nsight Compute first."
  exit 1
fi

if [ ! -x "$BIN_PATH" ]; then
  echo "binary not found: $BIN_PATH"
  echo "run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

set +e
ncu \
  --target-processes all \
  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum \
  --csv \
  --log-file "${OUT_BASE}.csv" \
  env BENCH_ITERS=1 "$BIN_PATH"
status=$?
set -e

if [ $status -ne 0 ]; then
  if awk '/ERR_NVGPUCTRPERM/ {found=1} END{exit found?0:1}' "${OUT_BASE}.csv"; then
    echo "NCU permission blocked by ERR_NVGPUCTRPERM."
    echo "Please enable GPU performance counter access, then rerun."
    exit 0
  fi
  echo "NCU profiling failed with exit code ${status}"
  exit $status
fi

echo "NCU report saved: ${OUT_BASE}.csv"
