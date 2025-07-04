#!/bin/bash
set -e

# Default parameters
BUILD_DIR=build
NP=8
ELTS=4194304
DTYPE=fp32
ITERS=10
CUDA_ARCH=90
TARGET=bench_custom_allreduce
ONLY_BUILD=0
RESET_BUILD=0

# Print usage information
usage() {
  echo "Usage: $0 [-n <num_procs>] [-e <elts>] [-d <dtype>] [-i <iters>] [-c <cuda_arch>] [-r] [-b]"
  echo
  echo "  -n  Number of MPI processes (default: $NP)"
  echo "  -e  Number of elements (default: $ELTS)"
  echo "  -d  Data type: fp32|fp16 (default: $DTYPE)"
  echo "  -i  Number of iterations (default: $ITERS)"
  echo "  -c  CUDA architecture (default: $CUDA_ARCH)"
  echo "  -b  Only build the project, do not run the benchmark"
  echo "  -r  Reset build directory (clean rebuild)"
  echo "  -h  Show this help message"
  exit 1
}

# Parse command-line options
while getopts ":n:e:d:i:c:rbh" opt; do
  case ${opt} in
    n ) NP="$OPTARG" ;;
    e ) ELTS="$OPTARG" ;;
    d ) DTYPE="$OPTARG" ;;
    i ) ITERS="$OPTARG" ;;
    c ) CUDA_ARCH="$OPTARG" ;;
    b ) ONLY_BUILD=1 ;;
    r ) RESET_BUILD=1 ;;
    h ) usage ;;
    * ) usage ;;
  esac
done

# Clean build directory if requested
if [[ "$RESET_BUILD" -eq 1 ]]; then
  echo ">>> Cleaning build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# Build project
echo ">>> Configuring and building project"
cmake -S . -B "$BUILD_DIR" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$BUILD_DIR" --parallel

# Run benchmark (if not only building)
if [[ "$ONLY_BUILD" -eq 0 ]]; then
  EXEC="./$BUILD_DIR/$TARGET"
  echo ">>> Running benchmark: mpirun -np $NP $EXEC --elts $ELTS --dtype $DTYPE --iters $ITERS"
  mpirun -np "$NP" "$EXEC" --elts "$ELTS" --dtype "$DTYPE" --iters "$ITERS"
fi
