import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
import numpy as np # For handling potential NaN/Inf

print("Generating benchmark plots using Python...")

# --- Configuration ---
LOG_BASE_DIR = os.path.join("logs", "benchmarks", "loop_types_benchmarks")
CSV_INPUT_DIR = os.path.join(LOG_BASE_DIR, "results")
CSV_INPUT_FILE = os.path.join(CSV_INPUT_DIR, "compiled_benchmarks.csv")
PLOT_OUTPUT_BASE_DIR = os.path.join("logs", "benchmarks", "loop_types_benchmarks")
METHODS_ORDER = ["Serial", "@turbo", "@floop", "ThreadsX"] # Consistent plotting order

# --- Check if input file exists ---
if not os.path.isfile(CSV_INPUT_FILE):
    print(f"Error: Compiled CSV file not found: {CSV_INPUT_FILE}", file=sys.stderr)
    print("Please run the log processing script first.", file=sys.stderr)
    sys.exit(1)

# --- Load Data ---
try:
    df = pd.read_csv(CSV_INPUT_FILE)
    print(f"Successfully loaded data from {CSV_INPUT_FILE}")
except Exception as e:
    print(f"Error loading CSV file: {e}", file=sys.stderr)
    sys.exit(1)

# --- Data Cleaning/Preparation ---
# Ensure 'threads' is integer type
df['threads'] = df['threads'].astype(int)
# Replace potential placeholder -1 in allocations with NaN (though we won't plot it)
df['allocations'] = df['allocations'].replace(-1, np.nan)
# Get unique thread counts for axis ticks
unique_threads = sorted(df['threads'].unique())

# --- Group by kernel ---
grouped = df.groupby('kernel')
print(f"Found data for kernels: {list(grouped.groups.keys())}")

# --- Generate Plots per Kernel ---
for kernel_name, group_df in grouped:
    n_particles = group_df['particles'].iloc[0] if not group_df.empty else "N/A"
    print(f"\nPlotting for kernel: {kernel_name} (N={n_particles})...")

    # Define plot output directory
    plot_dir = os.path.join(PLOT_OUTPUT_BASE_DIR, kernel_name)
    os.makedirs(plot_dir, exist_ok=True) # Ensure directory exists

    # --- Create Figure with 2 Subplots (Side-by-Side) ---
    # 1 row, 2 columns. Adjust figsize for better aspect ratio.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Kernel: {kernel_name} (N={n_particles})", fontsize=14)

    # Assign axes for clarity
    ax_time = axes[0]
    ax_mem = axes[1]

    # Use a consistent color cycle or define colors per method if needed
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(METHODS_ORDER)}

    # --- Configure Left Subplot (Time) ---
    ax_time.set_title("Median Time vs Threads")
    ax_time.set_xlabel("Number of Threads")
    ax_time.set_ylabel("Median Time (ms, log scale)")
    ax_time.set_yscale('log')
    ax_time.set_xticks(unique_threads)
    ax_time.set_xticklabels(unique_threads)
    ax_time.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # --- Configure Right Subplot (Memory) ---
    ax_mem.set_title("Memory vs Threads")
    ax_mem.set_xlabel("Number of Threads")
    ax_mem.set_ylabel("Memory Estimate (MiB)")
    # ax_mem.set_yscale('log') # Usually linear, uncomment if needed
    ax_mem.set_xticks(unique_threads)
    ax_mem.set_xticklabels(unique_threads)
    ax_mem.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    # Adjust memory y-axis limits slightly if linear
    if ax_mem.get_yscale() == 'linear':
        valid_mem_vals = group_df['memory_mib'].dropna()
        if not valid_mem_vals.empty:
            min_mem = valid_mem_vals.min()
            max_mem = valid_mem_vals.max()
            pad = 0.1 * (max_mem - min_mem) if max_mem > min_mem else 1.0 # Avoid zero padding
            ax_mem.set_ylim(bottom=max(0, min_mem - pad), top=max_mem + pad)
        else:
            ax_mem.set_ylim(bottom=0)


    # --- Populate Subplots ---
    for method in METHODS_ORDER:
        method_data = group_df[group_df['method'] == method].sort_values('threads')
        method_color = method_colors[method]

        if not method_data.empty:
            # Plot Time Data on ax_time
            valid_time_data = method_data.dropna(subset=['median_time_ms'])
            if not valid_time_data.empty:
                if method in ["Serial", "@turbo"] and all(valid_time_data['threads'] == 1):
                    ax_time.axhline(valid_time_data['median_time_ms'].iloc[0],
                                    label=f"{method}", # Simpler label for single legend
                                    color=method_color, linestyle='-', linewidth=1.5, alpha=0.9)
                else:
                    ax_time.plot(valid_time_data['threads'], valid_time_data['median_time_ms'],
                                 label=method, marker='o', markersize=5, linestyle='-',
                                 linewidth=1.5, color=method_color)

            # Plot Memory Data on ax_mem
            valid_mem_data = method_data.dropna(subset=['memory_mib'])
            if not valid_mem_data.empty:
                if method in ["Serial", "@turbo"] and all(valid_mem_data['threads'] == 1):
                    ax_mem.axhline(valid_mem_data['memory_mib'].iloc[0],
                                   label=f"{method}",
                                   color=method_color, linestyle='--', linewidth=1.5, alpha=0.9) # Dashed for memory
                else:
                    ax_mem.plot(valid_mem_data['threads'], valid_mem_data['memory_mib'],
                                label=method, marker='s', markersize=5, linestyle='--', # Dashed line
                                linewidth=1.5, color=method_color)


    # --- Add Legends ---
    ax_time.legend(loc='best')
    ax_mem.legend(loc='best')

    # --- Adjust Layout and Save ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout, leave space for suptitle
    plot_filename = os.path.join(plot_dir, f"{kernel_name}_{n_particles}_time_memory_subplot.png")
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"  Saved: {plot_filename}")
    except Exception as e:
        print(f"  Error saving subplot figure: {e}", file=sys.stderr)

    plt.close(fig) # Close plot to free memory

print("\nPlot generation finished.")