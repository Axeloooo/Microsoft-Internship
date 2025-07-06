#!/bin/bash

# Clean script for TRT AllReduce project

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
BUILD_DIR="$PROJECT_ROOT/build"
DATA_DIR="$PROJECT_ROOT/data"
PLOTS_DIR="$PROJECT_ROOT/images"

# Print usage information
usage() {
  echo "Usage: $0 [-a] [-b] [-d] [-p] [-h]"
  echo
  echo "  -a  Clean all (build, data, plots)"
  echo "  -b  Clean build directory only"
  echo "  -d  Clean data directory only"
  echo "  -p  Clean plots directory only"
  echo "  -h  Show this help message"
  echo
  echo "If no options are specified, only the build directory will be cleaned."
  exit 1
}

CLEAN_BUILD=0
CLEAN_DATA=0
CLEAN_PLOTS=0
CLEAN_ALL=0

# Parse command-line options
while getopts ":abdph" opt; do
  case ${opt} in
    a ) CLEAN_ALL=1 ;;
    b ) CLEAN_BUILD=1 ;;
    d ) CLEAN_DATA=1 ;;
    p ) CLEAN_PLOTS=1 ;;
    h ) usage ;;
    * ) usage ;;
  esac
done

# If no options specified, clean build by default
if [[ $# -eq 0 ]]; then
  CLEAN_BUILD=1
fi

# If clean all is specified, set all flags
if [[ "$CLEAN_ALL" -eq 1 ]]; then
  CLEAN_BUILD=1
  CLEAN_DATA=1
  CLEAN_PLOTS=1
fi

echo ">>> TRT AllReduce Cleanup"

# Clean build directory
if [[ "$CLEAN_BUILD" -eq 1 ]]; then
  if [[ -d "$BUILD_DIR" ]]; then
    echo ">>> Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  else
    echo ">>> Build directory does not exist: $BUILD_DIR"
  fi
fi

# Clean data directory
if [[ "$CLEAN_DATA" -eq 1 ]]; then
  if [[ -d "$DATA_DIR" ]]; then
    echo ">>> Cleaning data directory: $DATA_DIR"
    rm -rf "$DATA_DIR"
  else
    echo ">>> Data directory does not exist: $DATA_DIR"
  fi
fi

# Clean plots directory
if [[ "$CLEAN_PLOTS" -eq 1 ]]; then
  if [[ -d "$PLOTS_DIR" ]]; then
    echo ">>> Cleaning plots directory: $PLOTS_DIR"
    rm -rf "$PLOTS_DIR"
  else
    echo ">>> Plots directory does not exist: $PLOTS_DIR"
  fi
fi

echo ">>> Cleanup completed!"
