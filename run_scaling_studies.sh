#!/bin/bash
# This script demonstrates how to run the benchmark analysis tool

# Path to your logs directory (adjust as needed)
LOG_DIR="logs/mpi_run"
OUTPUT_CSV="benchmark_results.csv"
PLOT_DIR="scaling_plots"

# Make sure the script is executable
chmod +x scaling_studies.py

# Run the analysis
echo "Starting benchmark analysis..."
./scaling_studies.py --log-dir "$LOG_DIR" --output-csv "$OUTPUT_CSV" --plot-dir "$PLOT_DIR"

echo "Analysis complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "Plots saved to: $PLOT_DIR"