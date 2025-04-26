#!/bin/bash
# Master script to run log parsing and plotting for scaling analysis
# This script runs the full analysis pipeline on the benchmark results

set -e  # Exit on error

module load matplotlib

# Default paths
LOG_PARSER="./parse_benchmark_logs.sh"
PLOT_SCRIPT="./plot_scaling.py"
OUTPUT_DIR="plots"
CSV_OUTPUT="benchmark_results.csv"

# Print banner
echo "=========================================================="
echo "StochasticHaissinski Benchmark Analysis Pipeline"
echo "=========================================================="

# Check if required scripts exist
if [ ! -f "$LOG_PARSER" ]; then
  echo "ERROR: Log parser script not found at $LOG_PARSER"
  exit 1
fi

if [ ! -f "$PLOT_SCRIPT" ]; then
  echo "ERROR: Plot script not found at $PLOT_SCRIPT"
  exit 1
fi

# Make scripts executable
chmod +x "$LOG_PARSER"
chmod +x "$PLOT_SCRIPT"

# Step 1: Parse log files
echo "Step 1: Parsing benchmark log files..."
time "$LOG_PARSER"

if [ ! -f "$CSV_OUTPUT" ]; then
  echo "ERROR: Log parsing failed - $CSV_OUTPUT not created"
  exit 1
fi

echo "Log parsing complete. Results saved to $CSV_OUTPUT"
echo

# Step 2: Generate plots
echo "Step 2: Generating scaling plots..."
time python3 "$PLOT_SCRIPT" --input "$CSV_OUTPUT" --output-dir "$OUTPUT_DIR" --analyses all

# Check if plots were created
if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "WARNING: No plots were generated in $OUTPUT_DIR"
else
  echo "Plot generation complete. Plots saved to $OUTPUT_DIR/"
  echo "Generated $(find "$OUTPUT_DIR" -type f | wc -l) plot files"
fi

echo
echo "Analysis pipeline completed successfully!"
echo "=========================================================="
echo "Summary:"
echo "- Parsed logs and generated $CSV_OUTPUT"
echo "- Created plots in $OUTPUT_DIR/"
echo "=========================================================="

