#!/usr/bin/env bash
# Simple wrapper to run Nsight Compute (ncu) against this project's `main.py`.
# Writes output to `reports/ncu/` (creates directory if needed).

set -euo pipefail

OUTDIR=reports/ncu
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

REPORT_PREFIX=$(next_report_prefix "$OUTDIR" "ncu_report" ".ncu-rep")

NCU_TIMEOUT=${NCU_TIMEOUT:-}

# Default ncu args: profile all target processes and use the 'full' metrics set.
# You can pass additional args after `--` when invoking this script; they are forwarded to `python3 main.py`.
NCU_CMD=(/usr/local/NVIDIA-Nsight-Compute-2026.1/ncu -o "$REPORT_PREFIX")

# Prefer running under the project's `uv` virtualenv if `uv` is available.
if command -v uv >/dev/null 2>&1; then
	RUN_CMD=(uv run main.py)
else
	RUN_CMD=(python3 main.py)
fi

echo "Running: ${NCU_CMD[*]} -- ${RUN_CMD[*]} $*"
if [[ -n "$NCU_TIMEOUT" ]]; then
	if ! command -v timeout >/dev/null 2>&1; then
		echo "NCU_TIMEOUT is set, but the 'timeout' command is not available." >&2
		exit 1
	fi

	timeout --foreground --kill-after=30s "$NCU_TIMEOUT" "${NCU_CMD[@]}" -- "${RUN_CMD[@]}" "$@"
else
	"${NCU_CMD[@]}" -- "${RUN_CMD[@]}" "$@"
fi

echo "Nsight Compute report written to: ${REPORT_PREFIX}.ncu-rep"
