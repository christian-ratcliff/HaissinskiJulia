#!/bin/bash
# run_verifications.sh
# Script to automatically run verifications with multiple MPI ranks
# Usage: ./tests/run_verifications.sh [--turns N] [--particles N] [--ranks "1,2,4,8"]

set -e  # Exit on error

# Default values
TURNS=5000
PARTICLES=100000
RANKS=(1 2 4 8)
TOLERANCE=1e-2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --turns)
            TURNS="$2"
            shift 2
            ;;
        --particles)
            PARTICLES="$2"
            shift 2
            ;;
        --ranks)
            IFS=',' read -ra RANKS <<< "$2"
            shift 2
            ;;
        --tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================"
echo "StochasticHaissinski MPI Verification Suite"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo "Configuration:"
echo "  Turns: $TURNS"
echo "  Particles: $PARTICLES"
echo "  Ranks to test: ${RANKS[*]}"
echo "  Tolerance: $TOLERANCE"
echo "========================================================"
echo ""

log_dir="logs/verification_logs"
mkdir -p "$log_dir"
timestamp=$(date '+%Y%m%d_%H%M%S')
summary_log="$log_dir/verification_summary_${timestamp}.log"

echo "Starting verification tests..."
echo "Results will be saved to: $summary_log"
echo ""

# Initialize summary log
{
    echo "StochasticHaissinski MPI Verification Summary"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"
    echo "Configuration:"
    echo "  Turns: $TURNS"
    echo "  Particles: $PARTICLES"
    echo "  Ranks tested: ${RANKS[*]}"
    echo "  Tolerance: $TOLERANCE"
    echo "========================================================"
    echo ""
} > "$summary_log"

# Get reference simulation results (serial run)
echo "Running reference simulation (1 rank)..."
reference_log="$log_dir/reference_${timestamp}.log"
mpiexecjl -n 1 julia --project=. tests/verify_mpi.jl --turns $TURNS --particles $PARTICLES --tolerance $TOLERANCE | tee "$reference_log"

# Get reference values (for summary)
ref_sigma_E=$(grep -oP 'σ_E = \K[0-9.e+-]+' "$reference_log" | head -1)
ref_sigma_z=$(grep -oP 'σ_z = \K[0-9.e+-]+' "$reference_log" | head -1)
ref_E0=$(grep -oP 'E0 = \K[0-9.e+-]+' "$reference_log" | head -1)

# Add reference values to summary
{
    echo "Reference values (1 MPI rank):"
    echo "  σ_E = $ref_sigma_E"
    echo "  σ_z = $ref_sigma_z"
    echo "  E0 = $ref_E0"
    echo ""
    echo "Verification Results:"
    echo "--------------------------------------------------------"
} >> "$summary_log"

all_tests_passed=true

# Run tests for each rank configuration
for n_ranks in "${RANKS[@]}"; do
    # Skip rank 1 since we already ran it as reference
    if [ "$n_ranks" -eq 1 ]; then
        continue
    fi
    
    echo ""
    echo "Testing with $n_ranks ranks..."
    current_log="$log_dir/rank${n_ranks}_${timestamp}.log"
    
    # Run the verification
    mpiexecjl -n "$n_ranks" julia --project=. tests/verify_mpi.jl --turns $TURNS --particles $PARTICLES --tolerance $TOLERANCE | tee "$current_log"
    
    # Check if test passed
    if grep -q "Overall Result: PASS" "$current_log"; then
        result="PASS"
    else
        result="FAIL"
        all_tests_passed=false
    fi
    
    # Extract results for summary
    mpi_sigma_E=$(grep -oP 'σ_E = \K[0-9.e+-]+' "$current_log" | tail -1)
    mpi_sigma_z=$(grep -oP 'σ_z = \K[0-9.e+-]+' "$current_log" | tail -1)
    mpi_E0=$(grep -oP 'E0 = \K[0-9.e+-]+' "$current_log" | tail -1)
    
    # Extract differences
    rel_diff_sigma_E=$(grep -oP 'σ_E relative difference: \K[0-9.e+-]+' "$current_log")
    rel_diff_sigma_z=$(grep -oP 'σ_z relative difference: \K[0-9.e+-]+' "$current_log")
    rel_diff_E0=$(grep -oP 'E0 relative difference: \K[0-9.e+-]+' "$current_log")
    
    # Add to summary
    {
        echo "$n_ranks ranks: $result"
        echo "  Values:"
        echo "    σ_E = $mpi_sigma_E (diff: $rel_diff_sigma_E)"
        echo "    σ_z = $mpi_sigma_z (diff: $rel_diff_sigma_z)"
        echo "    E0 = $mpi_E0 (diff: $rel_diff_E0)"
        echo "--------------------------------------------------------"
    } >> "$summary_log"
done

# Add overall result to summary
{
    echo ""
    echo "Overall Verification Suite Result: $([ "$all_tests_passed" = true ] && echo "PASS" || echo "FAIL")"
    echo "Complete logs available in: $log_dir"
} >> "$summary_log"

echo ""
echo "Verification tests completed."
echo "Summary available in: $summary_log"
echo ""
echo "Overall result: $([ "$all_tests_passed" = true ] && echo "PASS" || echo "FAIL")"

# Exit with appropriate status code
[ "$all_tests_passed" = true ] && exit 0 || exit 1