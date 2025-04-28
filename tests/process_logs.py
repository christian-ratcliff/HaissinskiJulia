import os
import re
import pandas as pd
import sys
import warnings

print("Processing benchmark log files using Python...")

# --- Configuration ---
LOG_BASE_DIR = os.path.join("logs", "benchmarks", "loop_types_benchmarks")
CSV_OUTPUT_DIR = os.path.join(LOG_BASE_DIR, "results")
CSV_OUTPUT_FILE = os.path.join(CSV_OUTPUT_DIR, "compiled_benchmarks.csv")

# --- Helper Function to Parse Filename ---
def parse_log_filename(filepath):
    """Extracts kernel, particles, and threads from the log filepath."""
    try:
        filename = os.path.basename(filepath)
        # Regex to capture N, Threads from filename like "100000_particles_4_threads.jl"
        m = re.search(r"(\d+)_particles_(\d+)_threads\.jl", filename)
        if m:
            n_particles = int(m.group(1))
            n_threads = int(m.group(2))
            # Get kernel name from the parent directory name
            kernel_name = os.path.basename(os.path.dirname(filepath))
            return kernel_name, n_particles, n_threads
        else:
            warnings.warn(f"Could not parse filename pattern: {filename}")
            return None, None, None
    except Exception as e:
        warnings.warn(f"Error parsing filename {filepath}: {e}")
        return None, None, None

# --- Helper Function to Extract Metrics from a File ---
def extract_metrics_from_file(filepath):
    """Reads a log file and extracts metrics for each method."""
    kernel_name, n_particles, n_threads = parse_log_filename(filepath)
    if kernel_name is None:
        return [] # Skip if filename parsing failed

    extracted_data = []
    # Use the sanitized names used as variables in the log file
    methods_log_prefixes = {
        "Serial": "Serial",
        "@turbo": "_turbo",
        "@floop": "_floop",
        "ThreadsX": "ThreadsX"
    }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        warnings.warn(f"Could not read file {filepath}: {e}")
        return []

    for display_name, log_prefix in methods_log_prefixes.items():
        # Define regex patterns dynamically
        time_pattern = re.compile(rf"^\s*{log_prefix}_median_time_ms\s*=\s*([\d\.]+)")
        mem_pattern = re.compile(rf"^\s*{log_prefix}_memory_mib\s*=\s*([\d\.]+)")
        alloc_pattern = re.compile(rf"^\s*{log_prefix}_allocations\s*=\s*(\d+)")

        # Initialize metrics for this method
        metrics = {
            "median_time_ms": float('nan'),
            "memory_mib": float('nan'),
            "allocations": -1 # Use -1 or None to indicate not found
        }
        found_any = False

        # Search lines for metrics for the current method
        for line in lines:
            time_match = time_pattern.search(line)
            if time_match:
                try:
                    metrics["median_time_ms"] = float(time_match.group(1))
                    found_any = True
                except ValueError: pass # Ignore parsing errors
                continue # Optimization: assume one metric per line

            mem_match = mem_pattern.search(line)
            if mem_match:
                try:
                    metrics["memory_mib"] = float(mem_match.group(1))
                    found_any = True
                except ValueError: pass
                continue

            alloc_match = alloc_pattern.search(line)
            if alloc_match:
                try:
                    metrics["allocations"] = int(alloc_match.group(1))
                    found_any = True
                except ValueError: pass
                continue

        # Only add if we found at least one metric for this method in this file
        if found_any:
            extracted_data.append({
                "kernel": kernel_name,
                "method": display_name, # Use the user-friendly name
                "particles": n_particles,
                "threads": n_threads,
                "median_time_ms": metrics["median_time_ms"],
                "memory_mib": metrics["memory_mib"],
                "allocations": metrics["allocations"]
            })

    return extracted_data

# --- Main Processing Logic ---
all_results = []
if not os.path.isdir(LOG_BASE_DIR):
    print(f"Error: Log directory not found: {LOG_BASE_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"Scanning directory: {LOG_BASE_DIR}")
# Walk through the directory structure
for root, dirs, files in os.walk(LOG_BASE_DIR):
    for file in files:
        if file.endswith(".jl"): # Process only the .jl summary files
            filepath = os.path.join(root, file)
            # print(f"Processing: {filepath}") # Uncomment for verbose logging
            file_data = extract_metrics_from_file(filepath)
            if file_data: # Only extend if data was extracted
                all_results.extend(file_data)

if not all_results:
    print("Error: No benchmark data could be extracted from log files.", file=sys.stderr)
    print(f"Ensure '.jl' files exist in subdirectories of {LOG_BASE_DIR} and contain the expected metric lines.")
    sys.exit(1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(all_results)

# --- Save to CSV ---
# Ensure output directory exists
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

try:
    df.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"\nSuccessfully compiled benchmark data to: {CSV_OUTPUT_FILE}")
except Exception as e:
    print(f"Error writing CSV file '{CSV_OUTPUT_FILE}': {e}", file=sys.stderr)
    sys.exit(1)

print("Log processing finished.")