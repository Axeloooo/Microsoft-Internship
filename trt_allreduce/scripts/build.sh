#!/bin/bash

# Build script for TRT AllReduce project

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
BUILD_DIR="$PROJECT_ROOT/build"
CUDA_ARCH=90
RESET_BUILD=0

# Print usage information
usage() {
  echo "Usage: $0 [-c <cuda_arch>] [-r] [-h]"
  echo
  echo "  -c  CUDA architecture (default: $CUDA_ARCH)"
  echo "  -r  Reset build directory (clean rebuild)"
  echo "  -h  Show this help message"
  exit 1
}

# Parse command-line options
while getopts ":c:rh" opt; do
  case ${opt} in
    c ) CUDA_ARCH="$OPTARG" ;;
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
echo ">>> Project root: $PROJECT_ROOT"
echo ">>> Build directory: $BUILD_DIR"
echo ">>> CUDA architecture: $CUDA_ARCH"
cd "$PROJECT_ROOT"

cmake -S . -B "$BUILD_DIR" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$BUILD_DIR" --parallel

echo ">>> Build completed successfully!"
echo ">>> Executable: $BUILD_DIR/bench_custom_allreduce"
