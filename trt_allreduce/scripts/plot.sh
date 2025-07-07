#!/usr/bin/env bash
#
# Plot script for TRT AllReduce benchmark results
set -euo pipefail

# Locate this script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
DATA_DIR="$PROJECT_ROOT/data"
CSV_FILE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [-f <csv_file>] [-d <data_dir>] [-h]

  -f  Specific CSV file to plot (optional; looks in <data_dir> if no path)
  -d  Data directory containing CSV files (default: $DATA_DIR)
  -h  Show this help message
EOF
  exit 1
}

# Parse options
while getopts ":f:d:h" opt; do
  case "$opt" in
    f) CSV_FILE="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    h|*) usage ;;
  esac
done

# Avoid matplotlib config permission issues
export MPLCONFIGDIR="/tmp/matplotlib-$(id -u)"
mkdir -p "$MPLCONFIGDIR"

# Ensure python3 exists
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found in PATH" >&2
  exit 1
fi

# Locate the Python plotting script
PLOT_DIR="$PROJECT_ROOT/python"
PLOT_SCRIPT="$PLOT_DIR/plot_benchmark.py"
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Error: plot script not found at $PLOT_SCRIPT" >&2
  exit 1
fi

# (Optional) install requirements quietly
REQ="$PLOT_DIR/requirements.txt"
if [[ -f "$REQ" ]]; then
  echo ">>> Installing Python requirements..."
  python3 -m pip install --user --quiet -r "$REQ" || true
fi

echo "=================================="
echo "  TRT AllReduce Plotting"
echo "=================================="

# Build list of CSVs
if [[ -n "$CSV_FILE" ]]; then
  # if no path, look in DATA_DIR
  if [[ ! -f "$CSV_FILE" && -f "$DATA_DIR/$CSV_FILE" ]]; then
    CSV_FILE="$DATA_DIR/$CSV_FILE"
  fi
  if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE" >&2
    exit 1
  fi
  csvs=( "$CSV_FILE" )
else
  # all CSVs in DATA_DIR
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: data dir not found: $DATA_DIR" >&2
    exit 1
  fi
  shopt -s nullglob
  mapfile -t csvs < <(printf '%s\n' "$DATA_DIR"/*.csv)
  if (( ${#csvs[@]} == 0 )); then
    echo "Error: no CSV files in $DATA_DIR" >&2
    echo "Run the benchmark first: scripts/benchmark.sh" >&2
    exit 1
  fi
fi

# Change into the python directory so that `../images/` maps to project_root/images
cd "$PLOT_DIR"

for csv in "${csvs[@]}"; do
  echo ">>> Plotting: $csv"
  python3 "$PLOT_SCRIPT" --csv "$csv"
done

echo "=================================="
echo ">>> Done! Plots saved under: $PROJECT_ROOT/images"
