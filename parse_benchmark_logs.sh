#!/bin/bash
# Script to parse StochasticHaissinski benchmark logs and extract scaling data
# This script analyzes all log files and outputs a CSV with performance metrics

# Output file
OUTPUT_CSV="benchmark_results.csv"
LOG_DIR="logs/mpi_run"

# Create the output CSV with headers
echo "processes,threads,total_cores,particles,turns,time_sec,memory_bytes,gc_time_sec,allocations,total_flops,gflops_rate,flops_per_particle_per_turn" > $OUTPUT_CSV

# Function to extract numeric values from formatted strings
extract_number() {
  local string="$1"
  # Extract just the numeric part before the unit
  echo "$string" | grep -o -E '[0-9]+(\.[0-9]+)?' | head -1
}

# Function to convert time units to seconds
convert_to_seconds() {
  local time_str="$1"
  if [[ $time_str == *"ms"* ]]; then
    local val=$(extract_number "$time_str")
    echo "scale=9; $val / 1000" | bc
  elif [[ $time_str == *"μs"* ]]; then
    local val=$(extract_number "$time_str")
    echo "scale=9; $val / 1000000" | bc
  elif [[ $time_str == *"ns"* ]]; then
    local val=$(extract_number "$time_str")
    echo "scale=9; $val / 1000000000" | bc
  else
    # Assume seconds
    extract_number "$time_str"
  fi
}

# Function to convert memory units to bytes
convert_to_bytes() {
  local mem_str="$1"
  if [[ $mem_str == *"KiB"* ]]; then
    local val=$(extract_number "$mem_str")
    echo "scale=0; $val * 1024" | bc
  elif [[ $mem_str == *"MiB"* ]]; then
    local val=$(extract_number "$mem_str")
    echo "scale=0; $val * 1048576" | bc
  elif [[ $mem_str == *"GiB"* ]]; then
    local val=$(extract_number "$mem_str")
    echo "scale=0; $val * 1073741824" | bc
  elif [[ $mem_str == *"TiB"* ]]; then
    local val=$(extract_number "$mem_str")
    echo "scale=0; $val * 1099511627776" | bc
  else
    # Assume bytes
    extract_number "$mem_str"
  fi
}

# Process all log files
find "$LOG_DIR" -type f -name "*.log" | sort | while read log_file; do
  echo "Processing $log_file..."
  
  # Extract parameters from filename
  filename=$(basename "$log_file")
  
  # Extract processes, threads, turns and particles from the log file content
  processes=$(grep "mpi_comm_size" "$log_file" | awk -F'= ' '{print $2}')
  threads=$(grep "num_threads_per_process" "$log_file" | awk -F'= ' '{print $2}')
  turns=$(grep "n_turns" "$log_file" | awk -F'= ' '{print $2}')
  particles=$(grep "n_particles_global" "$log_file" | awk -F'= ' '{print $2}')
  total_cores=$((processes * threads))
  
  # Extract performance metrics
  time_str=$(grep "Max Median Time" "$log_file" | awk -F': ' '{print $2}')
  mem_str=$(grep "Sum of Median Memory Allocated" "$log_file" | awk -F': ' '{print $2}')
  gc_time_str=$(grep "Sum of Median GC Time" "$log_file" | sed 's/(.*)//g' | awk -F': ' '{print $2}')
  allocations=$(grep "Sum of Median Allocations" "$log_file" | awk -F': ' '{print $2}')
  
  # Extract FLOPS data if available
  total_flops_line=$(grep "Total aggregated FLOPs:" "$log_file")
  gflops_rate_line=$(grep "Aggregated GFLOPS rate" "$log_file")
  flops_per_particle_line=$(grep "FLOPs per particle per turn:" "$log_file")
  
  # Extract values or set to NA if not found
  if [[ -n "$total_flops_line" ]]; then
    total_flops=$(echo "$total_flops_line" | awk -F': ' '{print $2}' | sed 's/[^0-9.]//g')
  else
    total_flops="NA"
  fi
  
  if [[ -n "$gflops_rate_line" ]]; then
    gflops_rate=$(echo "$gflops_rate_line" | awk -F': ' '{print $2}' | sed 's/ GFLOPS//' | sed 's/[^0-9.]//g')
  else
    gflops_rate="NA"
  fi
  
  if [[ -n "$flops_per_particle_line" ]]; then
    flops_per_particle=$(echo "$flops_per_particle_line" | awk -F': ' '{print $2}' | sed 's/[^0-9.]//g')
  else
    flops_per_particle="NA"
  fi
  
  # Convert time and memory to base units
  time_sec=$(convert_to_seconds "$time_str")
  memory_bytes=$(convert_to_bytes "$mem_str")
  gc_time_sec=$(convert_to_seconds "$gc_time_str")
  
  # Write to CSV
  echo "$processes,$threads,$total_cores,$particles,$turns,$time_sec,$memory_bytes,$gc_time_sec,$allocations,$total_flops,$gflops_rate,$flops_per_particle" >> $OUTPUT_CSV
done

echo "Parsing complete. Results saved to $OUTPUT_CSV"