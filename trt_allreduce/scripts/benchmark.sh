#!/bin/bash

# Benchmark script for TRT AllReduce project

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
BUILD_DIR="$PROJECT_ROOT/build"
NP=8
DTYPE="fp32"
ITERS=1
STRATEGY="all"

# Print usage information
usage() {
  echo "Usage: $0 [-n <num_procs>] [-d <dtype>] [-i <iters>] [-h]"
  echo
  echo "  -n  Number of MPI processes (default: $NP)"
  echo "  -d  Data type: fp32|fp16 (default: $DTYPE)"
  echo "  -i  Number of iterations (default: $ITERS)"
  echo "  -s  Strategy: all|oneshot|twoshot|ub (default: $STRATEGY)"
  echo "  -h  Show this help message"
  echo
  echo "Note: Make sure to build the project first using scripts/build.sh"
  exit 1
}

# Parse command-line options
while getopts ":n:d:i:s:h" opt; do
  case ${opt} in
    n ) NP="$OPTARG" ;;
    d ) DTYPE="$OPTARG" ;;
    i ) ITERS="$OPTARG" ;;
    s ) STRATEGY="$OPTARG" ;;
    h ) usage ;;
    * ) usage ;;
  esac
done


# Validate number of processes
if ! [[ "$NP" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: Invalid number of processes '$NP'. Must be a positive integer."
  exit 1
fi

# Validate data type
if [[ "$DTYPE" != "fp32" && "$DTYPE" != "fp16" ]]; then
  echo "Error: Invalid data type '$DTYPE'. Must be 'fp32' or 'fp16'."
  exit 1
fi

# Validate number of iterations
if ! [[ "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: Invalid number of iterations '$ITERS'. Must be a positive integer."
  exit 1
fi

# Validate strategy
if [[ "$STRATEGY" != "all" && "$STRATEGY" != "oneshot" && "$STRATEGY" != "twoshot" && "$STRATEGY" != "ub" ]]; then
  echo "Error: Invalid strategy '$STRATEGY'. Must be 'all', 'oneshot', 'twoshot', or 'ub'."
  exit 1
fi

# Check if executable exists
EXEC="$BUILD_DIR/bench_trt_allreduce"
if [[ ! -f "$EXEC" ]]; then
  echo "Error: Executable not found: $EXEC"
  echo "Please build the project first using: scripts/build.sh"
  exit 1
fi

# Create data directory
mkdir -p "$PROJECT_ROOT/data"

echo "=================================="
echo "TRT AllReduce Benchmark"
echo "=================================="
echo "MPI processes: $NP"
echo "Data type: $DTYPE"
echo "Iterations: $ITERS"
echo "Strategy: $STRATEGY"
echo "Executable: $EXEC"
echo "=================================="

# Run benchmark
echo ">>> Running benchmark: mpirun -np $NP $EXEC --dtype $DTYPE --iters $ITERS --strat $STRATEGY"
cd "$PROJECT_ROOT"
mpirun -np "$NP" "$EXEC" --dtype "$DTYPE" --iters "$ITERS" --strat "$STRATEGY"

echo ""
echo ">>> Benchmark completed successfully!"
echo ">>> CSV files saved in: $PROJECT_ROOT/data/"
ls -la "$PROJECT_ROOT/data/"*.csv 2>/dev/null || echo "No CSV files found"
