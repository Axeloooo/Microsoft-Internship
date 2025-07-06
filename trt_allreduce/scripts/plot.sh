#!/bin/bash

# Plot script for TRT AllReduce benchmark results

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default parameters
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/plots"
CSV_FILE=""

# Print usage information
usage() {
  echo "Usage: $0 [-f <csv_file>] [-d <data_dir>] [-o <output_dir>] [-h]"
  echo
  echo "  -f  Specific CSV file to plot (optional)"
  echo "  -d  Data directory containing CSV files (default: $DATA_DIR)"
  echo "  -o  Output directory for plots (default: $OUTPUT_DIR)"
  echo "  -h  Show this help message"
  echo
  echo "Examples:"
  echo "  $0                                    # Plot all CSV files in data/"
  echo "  $0 -f data/benchmark_fp32.csv        # Plot specific CSV file"
  echo "  $0 -d /path/to/csv/files             # Plot all CSV files in custom directory"
  exit 1
}

# Parse command-line options
while getopts ":f:d:o:h" opt; do
  case ${opt} in
    f ) CSV_FILE="$OPTARG" ;;
    d ) DATA_DIR="$OPTARG" ;;
    o ) OUTPUT_DIR="$OPTARG" ;;
    h ) usage ;;
    * ) usage ;;
  esac
done

# Set matplotlib config directory to avoid permission issues
export MPLCONFIGDIR="/tmp/matplotlib-$(whoami)"
mkdir -p "$MPLCONFIGDIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if the Python plotting script exists
PLOT_SCRIPT="$PROJECT_ROOT/python/plot_benchmark.py"
if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "Error: Plot script not found: $PLOT_SCRIPT"
    exit 1
fi

# Install Python requirements if needed
REQUIREMENTS_FILE="$PROJECT_ROOT/python/requirements.txt"
if [[ -f "$REQUIREMENTS_FILE" ]]; then
    echo ">>> Installing Python requirements..."
    pip3 install -r "$REQUIREMENTS_FILE" --user --quiet || {
        echo "Warning: Could not install Python requirements"
        echo "You may need to install them manually:"
        echo "  pip3 install -r $REQUIREMENTS_FILE --user"
    }
fi

echo "=================================="
echo "TRT AllReduce Plotting"
echo "=================================="

# Check what to plot
if [[ -n "$CSV_FILE" ]]; then
    # Plot specific CSV file
    if [[ ! -f "$CSV_FILE" ]]; then
        echo "Error: CSV file not found: $CSV_FILE"
        exit 1
    fi
    echo ">>> Plotting specific file: $CSV_FILE"
    echo ">>> Output directory: $OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    python3 "$PLOT_SCRIPT" "$(dirname "$CSV_FILE")" --save-plots --output-dir "$OUTPUT_DIR" --file "$(basename "$CSV_FILE")"
else
    # Plot all CSV files in data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "Error: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Check if there are any CSV files
    if ! ls "$DATA_DIR"/*.csv &> /dev/null; then
        echo "Error: No CSV files found in: $DATA_DIR"
        echo "Run the benchmark first using: scripts/benchmark.sh"
        exit 1
    fi
    
    echo ">>> Plotting all CSV files in: $DATA_DIR"
    echo ">>> Output directory: $OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    python3 "$PLOT_SCRIPT" "$DATA_DIR" --save-plots --output-dir "$OUTPUT_DIR"
fi

echo "=================================="
echo ">>> Plotting completed successfully!"
if [[ -d "$OUTPUT_DIR" ]]; then
    echo ">>> Plots saved in: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "No plots found"
else
    echo ">>> No plots directory found"
fi

if [[ "$SAVE_PLOTS" -eq 1 ]]; then
  echo ">>> Plots saved to: $OUTPUT_DIR"
fi

echo ">>> Visualization complete!"
