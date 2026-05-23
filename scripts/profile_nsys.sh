#!/usr/bin/env bash
# Lightweight wrapper to run Nsight Systems (`nsys`) against this project's `main.py`.
# Output placed in `reports/nsys/` (created if missing).

set -euo pipefail

OUTDIR=reports/nsys
mkdir -p "$OUTDIR"

next_report_prefix() {
	local output_dir="$1"
	local base_name="$2"
	local extension="$3"
	local candidate="$output_dir/$base_name"
	local suffix=1

	while [[ -e "$candidate$extension" ]]; do
		candidate="$output_dir/${base_name}_$suffix"
		((suffix++))
	done

	printf '%s\n' "$candidate"
}

REPORT_PREFIX=$(next_report_prefix "$OUTDIR" "nsys_report" ".nsys-rep")

# Default nsys profile args (match README example). Pass additional args to python via script args.
NSYS_CMD=(nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=all --gpu-metrics-frequency=10000 --trace-fork-before-exec=true -o "$REPORT_PREFIX")

# Prefer running under the project's `uv` virtualenv if `uv` is available.
if command -v uv >/dev/null 2>&1; then
	RUN_CMD=(uv run main.py)
else
	RUN_CMD=(python3 main.py)
fi

echo "Running: ${NSYS_CMD[*]} -- ${RUN_CMD[*]} $*"
"${NSYS_CMD[@]}" -- "${RUN_CMD[@]}" "$@"

echo "Nsight Systems report written to: ${REPORT_PREFIX}.nsys-rep"
