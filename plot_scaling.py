#!/usr/bin/env python3
"""
Comprehensive Scaling Analysis for StochasticHaissinski Benchmark Results

This script performs a complete scaling analysis, including:
- Strong scaling analysis (fixed problem size, varying core count)
- Memory usage analysis (how memory scales with tasks)
- Weak scaling analysis (fixed work per processor, varying problem size and core count)
- Thread-to-thread speedup analysis (performance at different thread counts)
- Process vs thread scaling comparison (optimal parallelization strategy)

Usage:
    python comprehensive_scaling_analysis.py [--input FILENAME] [--output-dir DIRECTORY]
        [--analyses ANALYSIS_LIST] [--no-report]

Example:
    python comprehensive_scaling_analysis.py --input benchmark_results.csv --analyses strong,memory,weak
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import sys
from matplotlib.ticker import ScalarFormatter
from datetime import datetime

# Define colors for different counts (threads or processes)
COLORS = {
    1: '#1f77b4',  # blue
    2: '#ff7f0e',  # orange
    4: '#2ca02c',  # green
    8: '#d62728',  # red
    16: '#9467bd', # purple
    32: '#8c564b', # brown
    64: '#e377c2', # pink
    128: '#7f7f7f', # gray
    256: '#bcbd22', # olive
    512: '#17becf'  # cyan
}

#######################
# SETUP AND UTILITIES #
#######################

def setup_plots():
    """Set up plot style for consistent appearance"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

def bytes_to_unit(bytes_value, unit='MB'):
    """Convert bytes to a specified unit"""
    if unit.upper() == 'KB':
        return bytes_value / 1024
    elif unit.upper() == 'MB':
        return bytes_value / (1024 * 1024)
    elif unit.upper() == 'GB':
        return bytes_value / (1024 * 1024 * 1024)
    else:
        return bytes_value

def load_data(filename='benchmark_results.csv'):
    """Load benchmark results from CSV file and prepare for analysis"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file {filename} not found. Run parse_benchmark_logs.sh first.")
    
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # Handle NA values for numeric columns
    numeric_columns = ['time_sec', 'memory_bytes', 'gc_time_sec', 'total_flops', 'gflops_rate', 
                       'flops_per_particle_per_turn', 'allocations']
    for col in df.columns:
        if col in numeric_columns:
            # Convert string 'NA' to actual NaN
            df[col] = df[col].replace('NA', np.nan)
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert scientific notation to numeric for these specific columns
    for col in ['particles', 'turns']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate derived metrics
    df['particles_per_core'] = df['particles'] / df['total_cores']
    
    # Calculate memory per core and per particle
    if 'memory_bytes' in df.columns:
        df['memory_per_core'] = df['memory_bytes'] / df['total_cores']
        df['memory_per_particle'] = df['memory_bytes'] / df['particles']
    
    # Print data summary
    print(f"Loaded {len(df)} benchmark results")
    print(f"Found {df['particles'].nunique()} particle counts, {df['turns'].nunique()} turn counts")
    print(f"Process counts: {sorted(df['processes'].unique())}")
    print(f"Thread counts: {sorted(df['threads'].unique())}")
    
    # Calculate speedup relative to single core (if available)
    try:
        baseline_times = {}
        for particles in df['particles'].unique():
            for turns in df['turns'].unique():
                subset = df[(df['particles'] == particles) & 
                           (df['turns'] == turns) & 
                           (df['total_cores'] == df['total_cores'].min())]
                if not subset.empty and not subset['time_sec'].isna().all():
                    baseline_times[(particles, turns)] = subset['time_sec'].values[0]
        
        # Calculate speedup only if we have valid baseline times
        if baseline_times:
            df['speedup'] = df.apply(
                lambda row: baseline_times.get((row['particles'], row['turns']), np.nan) / row['time_sec'] 
                if row['time_sec'] > 0 and pd.notna(row['time_sec']) else np.nan, 
                axis=1
            )
            
            # Calculate parallel efficiency
            df['parallel_efficiency'] = df.apply(
                lambda row: row['speedup'] / row['total_cores'] 
                if pd.notna(row['speedup']) and row['total_cores'] > 0 else np.nan,
                axis=1
            )
        else:
            print("Warning: No valid baseline times found for speedup calculation")
            df['speedup'] = np.nan
            df['parallel_efficiency'] = np.nan
    except Exception as e:
        print(f"Warning: Couldn't calculate speedup/efficiency: {e}")
        df['speedup'] = np.nan
        df['parallel_efficiency'] = np.nan
    
    # Check for negative or zero values in columns that will be log-scaled
    for col in ['time_sec', 'speedup', 'gflops_rate']:
        if col in df.columns:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                print(f"Warning: Found {invalid_count} rows with invalid values (<=0) in column '{col}'")
    
    return df

#######################
# STRONG SCALING PLOTS #
#######################

def plot_strong_scaling(df, output_dir='plots/strong_scaling'):
    """Create strong scaling plots (fixed problem size, varying core count)"""
    print("Generating strong scaling plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each problem size
    for particles in df['particles'].unique():
        for turns in df['turns'].unique():
            subset = df[(df['particles'] == particles) & (df['turns'] == turns)]
            
            if len(subset) < 2:
                continue  # Skip if not enough data points
            
            # Check if we have valid time data for log scaling
            if subset['time_sec'].min() <= 0:
                print(f"Warning: Invalid time values for particles={particles}, turns={turns}. Skipping log-scale plots.")
                continue
            
            # Group by process/thread combinations with same total cores
            process_thread_groups = {}
            for _, row in subset.iterrows():
                cores = row['total_cores']
                if cores not in process_thread_groups:
                    process_thread_groups[cores] = []
                process_thread_groups[cores].append(row)
            
            # Plot 1: Execution Time vs Core Count
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each thread count
                thread_counts = sorted(subset['threads'].unique())
                for thread_count in thread_counts:
                    thread_data = subset[subset['threads'] == thread_count]
                    if not thread_data.empty:
                        # Check for positive values before plotting with log scale
                        if thread_data['time_sec'].min() > 0:
                            color = COLORS.get(thread_count, 'black')
                            plt.plot(thread_data['total_cores'], thread_data['time_sec'], 
                                    'o-', label=f'{thread_count} threads/proc', color=color, markersize=8)
                
                if plt.gca().get_lines():  # Check if any lines were actually plotted
                    plt.xscale('log', base=2)
                    plt.yscale('log', base=10)
                    plt.xlabel('Total Number of Cores')
                    plt.ylabel('Execution Time (seconds)')
                    plt.title(f'Strong Scaling: Time vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add ideal scaling line if we have enough data
                    min_cores = subset['total_cores'].min()
                    min_time = subset[subset['total_cores'] == min_cores]['time_sec'].values[0]
                    if min_time > 0:
                        x_ideal = np.array(sorted(subset['total_cores'].unique()))
                        y_ideal = min_time * (min_cores / x_ideal)
                        plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling', alpha=0.7)
                    
                    try:
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/strong_scaling_time_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                    except Exception as e:
                        print(f"Warning: Could not save plot for particles={particles}, turns={turns}: {e}")
                else:
                    print(f"Warning: No valid data to plot for particles={particles}, turns={turns}")
                
                plt.close()
            except Exception as e:
                print(f"Error creating plot for particles={particles}, turns={turns}: {e}")
                plt.close()
            
            # Plot 2: Speedup vs Core Count
            if 'speedup' in df.columns and not subset['speedup'].isna().all():
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Check if we have valid speedup data for log scaling
                    valid_data = False
                    
                    # Plot for each thread count
                    for thread_count in thread_counts:
                        thread_data = subset[(subset['threads'] == thread_count) & (subset['speedup'] > 0)]
                        if not thread_data.empty:
                            valid_data = True
                            color = COLORS.get(thread_count, 'black')
                            plt.plot(thread_data['total_cores'], thread_data['speedup'], 
                                    'o-', label=f'{thread_count} threads/proc', color=color, markersize=8)
                    
                    if valid_data:
                        plt.xscale('log', base=2)
                        plt.yscale('log', base=2)
                        plt.xlabel('Total Number of Cores')
                        plt.ylabel('Speedup')
                        plt.title(f'Strong Scaling: Speedup vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                        plt.grid(True, which="both", ls="-", alpha=0.2)
                        plt.legend()
                        
                        # Add ideal scaling line
                        if 'x_ideal' in locals() and min_cores > 0:
                            plt.plot(x_ideal, x_ideal/min_cores, 'k--', label='Ideal speedup', alpha=0.7)
                        
                        try:
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/strong_scaling_speedup_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                        except Exception as e:
                            print(f"Warning: Could not save speedup plot for particles={particles}, turns={turns}: {e}")
                    else:
                        print(f"Warning: No valid speedup data for particles={particles}, turns={turns}")
                    
                    plt.close()
                except Exception as e:
                    print(f"Error creating speedup plot for particles={particles}, turns={turns}: {e}")
                    plt.close()
            
            # Plot 3: Parallel Efficiency vs Core Count
            if 'parallel_efficiency' in df.columns and not subset['parallel_efficiency'].isna().all():
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Check if we have valid data
                    valid_data = False
                    
                    # Plot for each thread count
                    for thread_count in thread_counts:
                        thread_data = subset[(subset['threads'] == thread_count) & (subset['parallel_efficiency'].notna())]
                        if not thread_data.empty:
                            valid_data = True
                            color = COLORS.get(thread_count, 'black')
                            plt.plot(thread_data['total_cores'], thread_data['parallel_efficiency'], 
                                    'o-', label=f'{thread_count} threads/proc', color=color, markersize=8)
                    
                    if valid_data:
                        plt.xscale('log', base=2)
                        plt.ylim([0, 1.1])
                        plt.xlabel('Total Number of Cores')
                        plt.ylabel('Parallel Efficiency')
                        plt.title(f'Strong Scaling: Efficiency vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                        plt.grid(True, which="both", ls="-", alpha=0.2)
                        plt.legend()
                        
                        # Add ideal efficiency line
                        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal efficiency')
                        
                        try:
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/strong_scaling_efficiency_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                        except Exception as e:
                            print(f"Warning: Could not save efficiency plot for particles={particles}, turns={turns}: {e}")
                    else:
                        print(f"Warning: No valid efficiency data for particles={particles}, turns={turns}")
                    
                    plt.close()
                except Exception as e:
                    print(f"Error creating efficiency plot for particles={particles}, turns={turns}: {e}")
                    plt.close()
            
            # Plot 4: GFLOPS Rate vs Core Count (if available)
            if 'gflops_rate' in df.columns and not subset['gflops_rate'].isna().all():
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Check if we have valid GFLOPS data
                    valid_data = False
                    
                    # Plot for each thread count
                    for thread_count in thread_counts:
                        thread_data = subset[subset['threads'] == thread_count].copy()
                        if not thread_data.empty and 'gflops_rate' in thread_data.columns:
                            # Filter and convert to numeric
                            thread_data = thread_data[thread_data['gflops_rate'] != 'NA']
                            if not thread_data.empty:
                                thread_data.loc[:, 'gflops_rate'] = pd.to_numeric(thread_data['gflops_rate'], errors='coerce')
                                # Remove any NaN or non-positive values
                                thread_data = thread_data[thread_data['gflops_rate'] > 0]
                                if not thread_data.empty:
                                    valid_data = True
                                    color = COLORS.get(thread_count, 'black')
                                    plt.plot(thread_data['total_cores'], thread_data['gflops_rate'], 
                                            'o-', label=f'{thread_count} threads/proc', color=color, markersize=8)
                    
                    if valid_data:
                        plt.xscale('log', base=2)
                        plt.yscale('log', base=10)
                        plt.xlabel('Total Number of Cores')
                        plt.ylabel('GFLOPS Rate')
                        plt.title(f'Performance: GFLOPS Rate vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                        plt.grid(True, which="both", ls="-", alpha=0.2)
                        plt.legend()
                        
                        # Add ideal scaling line if we have enough data
                        try:
                            if len(subset) > 1:
                                subset_with_flops = subset[subset['gflops_rate'] != 'NA'].copy()
                                if not subset_with_flops.empty:
                                    subset_with_flops.loc[:, 'gflops_rate'] = pd.to_numeric(subset_with_flops['gflops_rate'], errors='coerce')
                                    subset_with_flops = subset_with_flops[subset_with_flops['gflops_rate'] > 0]
                                    if not subset_with_flops.empty:
                                        min_cores_flops = subset_with_flops['total_cores'].min()
                                        min_flops_rows = subset_with_flops[subset_with_flops['total_cores'] == min_cores_flops]
                                        if not min_flops_rows.empty:
                                            min_flops = min_flops_rows['gflops_rate'].values[0]
                                            x_ideal_flops = np.array(sorted(subset_with_flops['total_cores'].unique()))
                                            y_ideal_flops = min_flops * (x_ideal_flops / min_cores_flops)
                                            plt.plot(x_ideal_flops, y_ideal_flops, 'k--', label='Ideal scaling', alpha=0.7)
                        except Exception as e:
                            print(f"Warning: Could not add ideal scaling line for GFLOPS plot: {e}")
                        
                        try:
                            plt.tight_layout()
                            plt.savefig(f'{output_dir}/performance_gflops_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                        except Exception as e:
                            print(f"Warning: Could not save GFLOPS plot for particles={particles}, turns={turns}: {e}")
                    else:
                        print(f"Warning: No valid GFLOPS data for particles={particles}, turns={turns}")
                    
                    plt.close()
                except Exception as e:
                    print(f"Error creating GFLOPS plot for particles={particles}, turns={turns}: {e}")
                    plt.close()
    
    print(f"Strong scaling plots generated in {output_dir}/")

#######################
# MEMORY SCALING PLOTS #
#######################

def plot_memory_scaling(df, output_dir='plots/memory'):
    """Create plots showing memory usage vs core count"""
    print("Generating memory scaling plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if memory data is available
    if 'memory_bytes' not in df.columns or df['memory_bytes'].isna().all():
        print("Warning: No memory data available. Skipping memory scaling plots.")
        return
    
    # Iterate through each problem size
    for particles in df['particles'].unique():
        for turns in df['turns'].unique():
            subset = df[(df['particles'] == particles) & 
                       (df['turns'] == turns) & 
                       (~df['memory_bytes'].isna())]
            
            if len(subset) < 2:
                continue  # Skip if not enough data points
            
            # Total memory usage vs cores
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each thread count
                thread_counts = sorted(subset['threads'].unique())
                for thread_count in thread_counts:
                    thread_data = subset[subset['threads'] == thread_count]
                    if not thread_data.empty:
                        color = COLORS.get(thread_count, 'black')
                        plt.plot(thread_data['total_cores'], 
                                bytes_to_unit(thread_data['memory_bytes'], 'MB'), 
                                'o-', label=f'{thread_count} threads/proc', 
                                color=color, markersize=8)
                
                if plt.gca().get_lines():  # Check if any lines were actually plotted
                    plt.xscale('log', base=2)
                    plt.yscale('log', base=2)
                    plt.xlabel('Total Number of Cores')
                    plt.ylabel('Total Memory Usage (MB)')
                    plt.title(f'Memory Scaling: Total Memory vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add linear scaling reference
                    if len(subset) > 1:
                        min_cores = subset['total_cores'].min()
                        min_memory = subset[subset['total_cores'] == min_cores]['memory_bytes'].values[0]
                        x_ideal = np.array(sorted(subset['total_cores'].unique()))
                        y_ideal = min_memory * (x_ideal / min_cores)
                        plt.plot(x_ideal, bytes_to_unit(y_ideal, 'MB'), 'k--', 
                                label='Linear scaling', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/memory_total_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating memory plot for particles={particles}, turns={turns}: {e}")
                plt.close()
            
            # Memory per core vs cores
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each thread count
                for thread_count in thread_counts:
                    thread_data = subset[subset['threads'] == thread_count]
                    if not thread_data.empty:
                        color = COLORS.get(thread_count, 'black')
                        plt.plot(thread_data['total_cores'], 
                                bytes_to_unit(thread_data['memory_per_core'], 'MB'), 
                                'o-', label=f'{thread_count} threads/proc', 
                                color=color, markersize=8)
                
                if plt.gca().get_lines():  # Check if any lines were actually plotted
                    plt.xscale('log', base=2)
                    plt.xlabel('Total Number of Cores')
                    plt.ylabel('Memory Per Core (MB)')
                    plt.title(f'Memory Scaling: Memory Per Core vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add constant reference line (ideal would be constant memory per core)
                    if len(subset) > 1:
                        min_cores = subset['total_cores'].min()
                        min_memory_per_core = subset[subset['total_cores'] == min_cores]['memory_per_core'].values[0]
                        x_ideal = np.array(sorted(subset['total_cores'].unique()))
                        y_ideal = np.ones_like(x_ideal) * min_memory_per_core
                        plt.plot(x_ideal, bytes_to_unit(y_ideal, 'MB'), 'k--', 
                                label='Ideal (constant memory/core)', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/memory_per_core_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating memory per core plot: {e}")
                plt.close()
            
            # Memory efficiency (memory per particle) vs cores
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each thread count
                for thread_count in thread_counts:
                    thread_data = subset[subset['threads'] == thread_count]
                    if not thread_data.empty:
                        color = COLORS.get(thread_count, 'black')
                        # Convert to KB since memory per particle is likely small
                        plt.plot(thread_data['total_cores'], 
                                bytes_to_unit(thread_data['memory_per_particle'], 'KB'), 
                                'o-', label=f'{thread_count} threads/proc', 
                                color=color, markersize=8)
                
                if plt.gca().get_lines():  # Check if any lines were actually plotted
                    plt.xscale('log', base=2)
                    plt.xlabel('Total Number of Cores')
                    plt.ylabel('Memory Per Particle (KB)')
                    plt.title(f'Memory Efficiency: Memory Per Particle vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/memory_per_particle_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating memory per particle plot: {e}")
                plt.close()

def create_memory_heatmap(df, output_dir='plots/memory'):
    """Create heatmaps showing memory usage for different process/thread combinations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if memory data is available
    if 'memory_bytes' not in df.columns or df['memory_bytes'].isna().all():
        return
    
    # Iterate through each problem size
    for particles in df['particles'].unique():
        for turns in df['turns'].unique():
            subset = df[(df['particles'] == particles) & 
                       (df['turns'] == turns) & 
                       (~df['memory_bytes'].isna())]
            
            if len(subset) < 4:  # Need enough data points
                continue
                
            # Create heatmap data for memory usage
            processes = sorted(subset['processes'].unique())
            threads = sorted(subset['threads'].unique())
            
            # Initialize matrix for heatmap
            memory_matrix = np.zeros((len(processes), len(threads)))
            memory_matrix.fill(np.nan)  # Fill with NaN for missing data
            
            memory_per_core_matrix = np.zeros((len(processes), len(threads)))
            memory_per_core_matrix.fill(np.nan)
            
            # Fill matrix with memory values
            for i, p in enumerate(processes):
                for j, t in enumerate(threads):
                    data = subset[(subset['processes'] == p) & (subset['threads'] == t)]
                    if not data.empty and not data['memory_bytes'].isna().all():
                        memory_matrix[i, j] = bytes_to_unit(data['memory_bytes'].values[0], 'MB')
                        memory_per_core_matrix[i, j] = bytes_to_unit(data['memory_per_core'].values[0], 'MB')
            
            # Plot memory heatmap
            try:
                plt.figure(figsize=(12, 10))
                plt.pcolormesh(memory_matrix, cmap='viridis')
                
                # Set ticks and labels
                plt.xticks(np.arange(len(threads)) + 0.5, threads)
                plt.yticks(np.arange(len(processes)) + 0.5, processes)
                plt.xlabel('Threads per Process')
                plt.ylabel('Number of Processes')
                plt.title(f'Total Memory Usage Heatmap (MB)\nParticles={particles:.0e}, Turns={turns:.0e}')
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Memory Usage (MB)')
                
                # Add values to cells
                for i in range(len(processes)):
                    for j in range(len(threads)):
                        if not np.isnan(memory_matrix[i, j]):
                            plt.text(j + 0.5, i + 0.5, f'{memory_matrix[i, j]:.1f}',
                                   ha='center', va='center', 
                                   color='white' if memory_matrix[i, j] > np.nanmean(memory_matrix) else 'black')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_memory_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating memory heatmap: {e}")
                plt.close()
            
            # Plot memory per core heatmap
            try:
                plt.figure(figsize=(12, 10))
                plt.pcolormesh(memory_per_core_matrix, cmap='coolwarm')
                
                # Set ticks and labels
                plt.xticks(np.arange(len(threads)) + 0.5, threads)
                plt.yticks(np.arange(len(processes)) + 0.5, processes)
                plt.xlabel('Threads per Process')
                plt.ylabel('Number of Processes')
                plt.title(f'Memory Per Core Heatmap (MB)\nParticles={particles:.0e}, Turns={turns:.0e}')
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Memory Per Core (MB)')
                
                # Add values to cells
                for i in range(len(processes)):
                    for j in range(len(threads)):
                        if not np.isnan(memory_per_core_matrix[i, j]):
                            plt.text(j + 0.5, i + 0.5, f'{memory_per_core_matrix[i, j]:.1f}',
                                   ha='center', va='center', 
                                   color='white' if memory_per_core_matrix[i, j] > np.nanmean(memory_per_core_matrix) else 'black')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_memory_per_core_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating memory per core heatmap: {e}")
                plt.close()

#######################
# WEAK SCALING PLOTS #
#######################

def identify_weak_scaling_sets(df):
    """Identify sets of runs that maintain the same particles per core ratio"""
    # Round particles_per_core to nearest power of 10 to group similar ratios
    df['particles_per_core_group'] = df['particles_per_core'].apply(
        lambda x: 10 ** round(np.log10(x))
    )
    
    # Group by particles_per_core_group and turn count
    weak_scaling_groups = []
    for turns in df['turns'].unique():
        # Get all particles_per_core groups for this turn count
        turn_df = df[df['turns'] == turns]
        for ppc_group in turn_df['particles_per_core_group'].unique():
            # Find all runs with this particles_per_core group and turn count
            group_df = turn_df[turn_df['particles_per_core_group'] == ppc_group]
            
            # Only consider groups with multiple core counts
            if len(group_df['total_cores'].unique()) > 1:
                weak_scaling_groups.append({
                    'turns': turns,
                    'particles_per_core': ppc_group,
                    'data': group_df
                })
    
    return weak_scaling_groups

def plot_weak_scaling(df, output_dir='plots/weak_scaling'):
    """Create weak scaling plots"""
    print("Generating weak scaling plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify weak scaling groups
    weak_scaling_groups = identify_weak_scaling_sets(df)
    
    if not weak_scaling_groups:
        print("No valid weak scaling groups found.")
        return
    
    print(f"Found {len(weak_scaling_groups)} weak scaling groups")
    
    # Plot for each weak scaling group
    for i, group in enumerate(weak_scaling_groups):
        turns = group['turns']
        ppc = group['particles_per_core']
        data = group['data']
        
        # Skip groups with less than 3 data points
        if len(data) < 3:
            continue
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Group by thread count
            thread_counts = sorted(data['threads'].unique())
            for thread_count in thread_counts:
                thread_data = data[data['threads'] == thread_count]
                if len(thread_data) > 1:  # Need at least 2 points to make a line
                    # Sort by total_cores
                    thread_data = thread_data.sort_values('total_cores')
                    color = COLORS.get(thread_count, 'black')
                    plt.plot(thread_data['total_cores'], thread_data['time_sec'], 
                            'o-', label=f'{thread_count} threads/proc', 
                            color=color, markersize=8)
            
            # Only continue if we plotted at least one line
            if plt.gca().get_lines():
                plt.xscale('log', base=2)
                if all(data['time_sec'] > 0):
                    plt.yscale('log', base=10)
                
                plt.xlabel('Total Number of Cores')
                plt.ylabel('Execution Time (seconds)')
                plt.title(f'Weak Scaling: Time vs Cores\nTurns={turns:.0e}, Particles/Core≈{ppc:.0e}')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.legend()
                
                # Add ideal scaling line (horizontal line for weak scaling)
                min_cores = data['total_cores'].min()
                min_time = data[data['total_cores'] == min_cores]['time_sec'].values[0]
                x_ideal = np.array(sorted(data['total_cores'].unique()))
                y_ideal = np.ones_like(x_ideal) * min_time
                plt.plot(x_ideal, y_ideal, 'k--', label='Ideal (constant time)', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/weak_scaling_t{turns:.0e}_ppc{ppc:.0e}.png', dpi=300)
            
            plt.close()
        except Exception as e:
            print(f"Error creating weak scaling plot: {e}")
            plt.close()
        
        # Create weak scaling efficiency plot
        try:
            plt.figure(figsize=(12, 8))
            
            # Compute weak scaling efficiency
            data = data.copy()
            min_cores = data['total_cores'].min()
            min_time = data[data['total_cores'] == min_cores]['time_sec'].values[0]
            
            # Weak scaling efficiency = T1 / Tn (where Tn is time for n cores)
            # Ideal value is 1.0 (constant time regardless of core count)
            data['weak_efficiency'] = min_time / data['time_sec']
            
            # Group by thread count
            for thread_count in thread_counts:
                thread_data = data[data['threads'] == thread_count]
                if len(thread_data) > 1:
                    thread_data = thread_data.sort_values('total_cores')
                    color = COLORS.get(thread_count, 'black')
                    plt.plot(thread_data['total_cores'], thread_data['weak_efficiency'], 
                            'o-', label=f'{thread_count} threads/proc', 
                            color=color, markersize=8)
            
            # Only continue if we plotted at least one line
            if plt.gca().get_lines():
                plt.xscale('log', base=2)
                plt.ylim([0, 1.5])  # Efficiency should be near 1.0 for ideal scaling
                
                plt.xlabel('Total Number of Cores')
                plt.ylabel('Weak Scaling Efficiency')
                plt.title(f'Weak Scaling Efficiency vs Cores\nTurns={turns:.0e}, Particles/Core≈{ppc:.0e}')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.legend()
                
                # Add ideal efficiency line (horizontal line at 1.0)
                plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal efficiency')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/weak_efficiency_t{turns:.0e}_ppc{ppc:.0e}.png', dpi=300)
            
            plt.close()
        except Exception as e:
            print(f"Error creating weak scaling efficiency plot: {e}")
            plt.close()

def plot_weak_scaling_overview(df, output_dir='plots/weak_scaling'):
    """Create overview plots showing weak scaling trends across all data"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a scatter plot of execution time vs particles per core
        plt.figure(figsize=(12, 8))
        
        # Group by core count
        core_groups = {}
        for cores in sorted(df['total_cores'].unique()):
            if cores in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:  # Select powers of 2
                core_data = df[df['total_cores'] == cores]
                if not core_data.empty:
                    core_groups[cores] = core_data
        
        # Plot each core group
        for cores, data in core_groups.items():
            # Only plot if we have valid data
            if not data.empty and 'particles_per_core' in data.columns and 'time_sec' in data.columns:
                valid_data = data[(data['particles_per_core'] > 0) & (data['time_sec'] > 0)]
                if not valid_data.empty:
                    plt.scatter(valid_data['particles_per_core'], valid_data['time_sec'], 
                              label=f'{cores} cores', s=50, alpha=0.7)
        
        if plt.gca().get_children():  # Check if any data was plotted
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)
            plt.xlabel('Particles per Core')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Weak Scaling Overview: Time vs Particles per Core')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/weak_scaling_overview.png', dpi=300)
        
        plt.close()
    except Exception as e:
        print(f"Error creating weak scaling overview plot: {e}")
        plt.close()
    
    try:
        # Create a heatmap showing average time per particle across core/particle combinations
        plt.figure(figsize=(12, 10))
        
        # Get unique core counts and particle counts
        core_counts = sorted([c for c in df['total_cores'].unique() if c in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]])
        particle_counts = sorted(df['particles'].unique())
        
        # Initialize matrix for heatmap
        time_per_particle_matrix = np.zeros((len(particle_counts), len(core_counts)))
        time_per_particle_matrix.fill(np.nan)  # Fill with NaN for missing data
        
        # Fill matrix
        for i, particles in enumerate(particle_counts):
            for j, cores in enumerate(core_counts):
                data = df[(df['particles'] == particles) & (df['total_cores'] == cores)]
                if not data.empty and not data['time_sec'].isna().all():
                    # Calculate time per particle (lower is better)
                    time_per_particle = data['time_sec'].mean() / particles
                    time_per_particle_matrix[i, j] = time_per_particle * 1e9  # Convert to nanoseconds for readability
        
        # Plot heatmap
        plt.pcolormesh(time_per_particle_matrix, cmap='viridis_r')  # Reversed so darker = better
        
        # Set ticks and labels
        plt.xticks(np.arange(len(core_counts)) + 0.5, core_counts)
        plt.yticks(np.arange(len(particle_counts)) + 0.5, [f"{p:.0e}" for p in particle_counts])
        plt.xlabel('Total Cores')
        plt.ylabel('Total Particles')
        plt.title('Time per Particle (ns) - Lower is Better')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Time per Particle (ns)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_per_particle_heatmap.png', dpi=300)
        
        plt.close()
    except Exception as e:
        print(f"Error creating time per particle heatmap: {e}")
        plt.close()
    
    print(f"Weak scaling plots generated in {output_dir}/")

#########################
# THREAD SPEEDUP PLOTS #
#########################

def calculate_thread_speedup(df):
    """Calculate thread speedup metrics for each process count"""
    # Group by process count, particle count, and turn count
    # For each group, calculate speedup relative to single thread
    
    # Create a copy of the dataframe to avoid modifying the original
    thread_df = df.copy()
    
    # Initialize the thread_speedup column
    thread_df['thread_speedup'] = np.nan
    
    # Process each group separately
    for particles in thread_df['particles'].unique():
        for turns in thread_df['turns'].unique():
            for processes in thread_df['processes'].unique():
                # Get data for this specific configuration
                group = thread_df[(thread_df['particles'] == particles) & 
                                  (thread_df['turns'] == turns) & 
                                  (thread_df['processes'] == processes)]
                
                # Skip if no data or only one thread count
                if len(group) <= 1:
                    continue
                
                # Find the single-thread time for this process count
                single_thread = group[group['threads'] == 1]
                if single_thread.empty or single_thread['time_sec'].isna().all():
                    # Try the lowest thread count if no single-thread data
                    min_threads = group['threads'].min()
                    single_thread = group[group['threads'] == min_threads]
                    if single_thread.empty or single_thread['time_sec'].isna().all():
                        continue
                
                base_time = single_thread['time_sec'].values[0]
                
                # Skip if base time is invalid
                if base_time <= 0 or np.isnan(base_time):
                    continue
                
                # Calculate speedup for each thread count in this group
                for idx, row in group.iterrows():
                    if row['time_sec'] > 0 and not np.isnan(row['time_sec']):
                        thread_df.loc[idx, 'thread_speedup'] = base_time / row['time_sec']
    
    return thread_df

def plot_thread_speedup(df, output_dir='plots/thread_speedup'):
    """Create plots showing thread speedup for different process counts"""
    print("Generating thread speedup plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate thread speedup
    thread_df = calculate_thread_speedup(df)
    
    # Iterate through each problem size
    for particles in thread_df['particles'].unique():
        for turns in thread_df['turns'].unique():
            subset = thread_df[(thread_df['particles'] == particles) & 
                              (thread_df['turns'] == turns)]
            
            if subset['thread_speedup'].isna().all():
                continue  # Skip if no valid speedup data
            
            # Thread count vs. execution time grouped by process count
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each process count
                process_counts = sorted(subset['processes'].unique())
                for process_count in process_counts:
                    process_data = subset[subset['processes'] == process_count]
                    if len(process_data) > 1:  # Need at least 2 points to make a line
                        # Sort by thread count
                        process_data = process_data.sort_values('threads')
                        if process_data['time_sec'].min() > 0:  # Check for positive values
                            color = COLORS.get(process_count, 'black')
                            plt.plot(process_data['threads'], process_data['time_sec'], 
                                   'o-', label=f'{process_count} processes', 
                                   color=color, markersize=8)
                
                # Only continue if we plotted at least one line
                if plt.gca().get_lines():
                    plt.xscale('log', base=2)
                    plt.yscale('log', base=10)
                    plt.xlabel('Number of Threads per Process')
                    plt.ylabel('Execution Time (seconds)')
                    plt.title(f'Thread Scaling: Time vs Thread Count\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/thread_time_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating thread scaling time plot: {e}")
                plt.close()
            
            # Thread count vs. speedup grouped by process count
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot for each process count
                for process_count in process_counts:
                    process_data = subset[(subset['processes'] == process_count) & 
                                        (subset['thread_speedup'].notna())]
                    if len(process_data) > 1:
                        # Sort by thread count
                        process_data = process_data.sort_values('threads')
                        color = COLORS.get(process_count, 'black')
                        plt.plot(process_data['threads'], process_data['thread_speedup'], 
                               'o-', label=f'{process_count} processes', 
                               color=color, markersize=8)
                
                # Only continue if we plotted at least one line
                if plt.gca().get_lines():
                    plt.xscale('log', base=2)
                    plt.xlabel('Number of Threads per Process')
                    plt.ylabel('Thread Speedup')
                    plt.title(f'Thread Speedup vs Thread Count\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add ideal thread scaling line
                    max_threads = subset['threads'].max()
                    x_ideal = np.array([1, max_threads])
                    y_ideal = x_ideal
                    plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/thread_speedup_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating thread speedup plot: {e}")
                plt.close()
            
            # Thread efficiency (speedup / thread count)
            try:
                plt.figure(figsize=(12, 8))
                
                for process_count in process_counts:
                    process_data = subset[(subset['processes'] == process_count) & 
                                        (subset['thread_speedup'].notna())]
                    if len(process_data) > 1:
                        # Calculate thread efficiency and sort by thread count
                        process_data = process_data.copy()
                        process_data['thread_efficiency'] = process_data['thread_speedup'] / process_data['threads']
                        process_data = process_data.sort_values('threads')
                        
                        color = COLORS.get(process_count, 'black')
                        plt.plot(process_data['threads'], process_data['thread_efficiency'], 
                               'o-', label=f'{process_count} processes', 
                               color=color, markersize=8)
                
                # Only continue if we plotted at least one line
                if plt.gca().get_lines():
                    plt.xscale('log', base=2)
                    plt.ylim([0, 1.2])  # Efficiency should be between 0 and 1
                    plt.xlabel('Number of Threads per Process')
                    plt.ylabel('Thread Efficiency (Speedup / Thread Count)')
                    plt.title(f'Thread Efficiency vs Thread Count\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add ideal efficiency line (horizontal line at 1.0)
                    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal efficiency')
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/thread_efficiency_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating thread efficiency plot: {e}")
                plt.close()

def create_speedup_heatmap(df, output_dir='plots/thread_speedup'):
    """Create heatmaps showing thread speedup for different process/thread combinations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate thread speedup
    thread_df = calculate_thread_speedup(df)
    
    # Iterate through each problem size
    for particles in thread_df['particles'].unique():
        for turns in thread_df['turns'].unique():
            subset = thread_df[(thread_df['particles'] == particles) & 
                              (thread_df['turns'] == turns) & 
                              (thread_df['thread_speedup'].notna())]
            
            if len(subset) < 4:  # Need enough data points
                continue
            
            try:
                plt.figure(figsize=(12, 10))
                
                # Get the unique process and thread counts
                processes = sorted(subset['processes'].unique())
                threads = sorted(subset['threads'].unique())
                
                # Initialize matrix for heatmap
                speedup_matrix = np.zeros((len(processes), len(threads)))
                speedup_matrix.fill(np.nan)  # Fill with NaN for missing data
                
                # Fill matrix with speedup values
                for i, p in enumerate(processes):
                    for j, t in enumerate(threads):
                        data = subset[(subset['processes'] == p) & (subset['threads'] == t)]
                        if not data.empty and not data['thread_speedup'].isna().all():
                            speedup_matrix[i, j] = data['thread_speedup'].values[0]
                
                # Plot heatmap
                plt.pcolormesh(speedup_matrix, cmap='viridis')
                
                # Set ticks and labels
                plt.xticks(np.arange(len(threads)) + 0.5, threads)
                plt.yticks(np.arange(len(processes)) + 0.5, processes)
                plt.xlabel('Threads per Process')
                plt.ylabel('Number of Processes')
                plt.title(f'Thread Speedup Heatmap\nParticles={particles:.0e}, Turns={turns:.0e}')
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Thread Speedup')
                
                # Add speedup values to cells
                for i in range(len(processes)):
                    for j in range(len(threads)):
                        if not np.isnan(speedup_matrix[i, j]):
                            plt.text(j + 0.5, i + 0.5, f'{speedup_matrix[i, j]:.2f}',
                                   ha='center', va='center', 
                                   color='white' if speedup_matrix[i, j] < np.nanmean(speedup_matrix) else 'black')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/heatmap_thread_speedup_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating thread speedup heatmap: {e}")
                plt.close()
    
    print(f"Thread speedup plots generated in {output_dir}/")

#################################
# THREAD VS PROCESS COMPARISON #
#################################

def plot_thread_vs_process(df, output_dir='plots/thread_vs_process'):
    """Create comparison plots between process and thread scaling"""
    print("Generating thread vs process comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each problem size
    for particles in df['particles'].unique():
        for turns in df['turns'].unique():
            subset = df[(df['particles'] == particles) & (df['turns'] == turns)]
            
            # Process scaling (1 thread per process)
            process_scaling = subset[subset['threads'] == 1].copy()
            process_scaling = process_scaling[process_scaling['time_sec'] > 0]
            process_scaling = process_scaling.sort_values('processes')
            
            # Thread scaling (1 process, multiple threads)
            thread_scaling = subset[subset['processes'] == 1].copy()
            thread_scaling = thread_scaling[thread_scaling['time_sec'] > 0]
            thread_scaling = thread_scaling.sort_values('threads')
            
            # Skip if we don't have enough data
            if len(process_scaling) < 2 and len(thread_scaling) < 2:
                continue
            
            # Create plot
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot process scaling
                if len(process_scaling) >= 2:
                    plt.plot(process_scaling['total_cores'], process_scaling['time_sec'], 
                            'bo-', label='Process scaling (threads=1)', markersize=8)
                
                # Plot thread scaling
                if len(thread_scaling) >= 2:
                    plt.plot(thread_scaling['total_cores'], thread_scaling['time_sec'], 
                            'ro-', label='Thread scaling (processes=1)', markersize=8)
                
                # Plot hybrid combinations (selected examples)
                hybrid_combinations = [
                    (2, 2), (2, 4), (2, 8), (4, 2), (4, 4), 
                    (8, 2), (8, 4), (16, 2), (16, 4), (32, 2)
                ]
                
                for proc, thread in hybrid_combinations:
                    hybrid_point = subset[(subset['processes'] == proc) & (subset['threads'] == thread)]
                    if not hybrid_point.empty and hybrid_point['time_sec'].values[0] > 0:
                        plt.plot(hybrid_point['total_cores'], hybrid_point['time_sec'], 'go', 
                                label=f'{proc}p×{thread}t', markersize=6, alpha=0.7)
                
                # Format plot
                plt.xlabel('Total Number of Cores')
                plt.ylabel('Execution Time (seconds)')
                plt.title(f'Process vs Thread Scaling\nParticles={particles:.0e}, Turns={turns:.0e}')
                plt.grid(True, alpha=0.2)
                
                # Only use log scale if all values are positive
                if np.all(subset['time_sec'] > 0):
                    plt.xscale('log', base=2)
                    plt.yscale('log', base=10)
                
                # Add ideal scaling line
                if len(process_scaling) >= 2 or len(thread_scaling) >= 2:
                    combined_cores = np.concatenate([process_scaling['total_cores'].values, 
                                                  thread_scaling['total_cores'].values])
                    combined_times = np.concatenate([process_scaling['time_sec'].values, 
                                                  thread_scaling['time_sec'].values])
                    min_cores_idx = np.argmin(combined_cores)
                    min_cores = combined_cores[min_cores_idx]
                    min_time = combined_times[min_cores_idx]
                    
                    x_ideal = np.array(sorted(subset['total_cores'].unique()))
                    y_ideal = min_time * (min_cores / x_ideal)
                    plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling', alpha=0.7)
                
                # Add legend with reasonable size
                plt.legend(fontsize=10, loc='best')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/proc_vs_thread_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating process vs thread plot: {e}")
                plt.close()
    
    print(f"Thread vs process comparison plots generated in {output_dir}/")

def plot_combined_scaling(df, output_dir='plots/combined_scaling'):
    """Create plot comparing thread-only, process-only, and combined scaling"""
    print("Generating combined scaling plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each problem size
    for particles in df['particles'].unique():
        for turns in df['turns'].unique():
            subset = df[(df['particles'] == particles) & (df['turns'] == turns)]
            
            if len(subset) < 4:  # Need enough data
                continue
            
            try:
                plt.figure(figsize=(12, 8))
                
                # Extract scaling data for different approaches
                
                # 1. Thread-only scaling (processes=1, varying threads)
                thread_only = subset[subset['processes'] == 1].sort_values('threads')
                if not thread_only.empty and thread_only['time_sec'].min() > 0:
                    plt.plot(thread_only['total_cores'], thread_only['time_sec'], 
                           'b-o', label='Thread-only scaling', linewidth=2, markersize=8)
                
                # 2. Process-only scaling (threads=1, varying processes)
                process_only = subset[subset['threads'] == 1].sort_values('processes')
                if not process_only.empty and process_only['time_sec'].min() > 0:
                    plt.plot(process_only['total_cores'], process_only['time_sec'], 
                           'r-o', label='Process-only scaling', linewidth=2, markersize=8)
                
                # 3. Balanced scaling (threads ≈ processes)
                balanced = subset[abs(subset['threads'] - subset['processes']) <= 1]
                if not balanced.empty and balanced['time_sec'].min() > 0:
                    balanced = balanced.sort_values('total_cores')
                    plt.plot(balanced['total_cores'], balanced['time_sec'], 
                           'g-o', label='Balanced scaling', linewidth=2, markersize=8)
                
                # 4. Optimal scaling (find the fastest configuration for each core count)
                optimal_times = []
                core_counts = sorted(subset['total_cores'].unique())
                for cores in core_counts:
                    core_data = subset[subset['total_cores'] == cores]
                    if not core_data.empty and not core_data['time_sec'].isna().all():
                        min_time = core_data['time_sec'].min()
                        if min_time > 0:
                            optimal_times.append((cores, min_time))
                
                if optimal_times:
                    opt_cores, opt_times = zip(*optimal_times)
                    plt.plot(opt_cores, opt_times, 'k-*', label='Optimal configuration', 
                           linewidth=2, markersize=10)
                
                # Only continue if we plotted at least one line
                if plt.gca().get_lines():
                    plt.xscale('log', base=2)
                    plt.yscale('log', base=10)
                    plt.xlabel('Total Number of Cores')
                    plt.ylabel('Execution Time (seconds)')
                    plt.title(f'Scaling Comparison: Time vs Cores\nParticles={particles:.0e}, Turns={turns:.0e}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend()
                    
                    # Add ideal scaling line if we have enough data
                    if len(optimal_times) > 1:
                        min_cores = optimal_times[0][0]
                        min_time = optimal_times[0][1]
                        x_ideal = np.array([min_cores] + list(opt_cores))
                        y_ideal = min_time * (min_cores / x_ideal)
                        plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/scaling_comparison_p{particles:.0e}_t{turns:.0e}.png', dpi=300)
                
                plt.close()
            except Exception as e:
                print(f"Error creating scaling comparison plot: {e}")
                plt.close()
    
    print(f"Combined scaling plots generated in {output_dir}/")

#######################
# MAIN FUNCTION #
#######################

def main():
    """Main function to run the comprehensive scaling analysis"""
    parser = argparse.ArgumentParser(
        description='Comprehensive scaling analysis for StochasticHaissinski benchmarks'
    )
    parser.add_argument('--input', default='benchmark_results.csv', 
                        help='Input CSV file with benchmark results')
    parser.add_argument('--output-dir', default='plots', 
                        help='Directory to save the plots')
    parser.add_argument('--analyses', default='all',
                        help='Comma-separated list of analyses to run (strong,memory,weak,thread,process,combined,all)')
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Setup
    setup_plots()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the data
        df = load_data(args.input)
        
        # Determine which analyses to run
        analyses = args.analyses.lower().split(',')
        run_all = 'all' in analyses
        
        # Run selected analyses
        if run_all or 'strong' in analyses:
            strong_dir = os.path.join(args.output_dir, 'strong_scaling')
            plot_strong_scaling(df, strong_dir)
        
        if run_all or 'memory' in analyses:
            memory_dir = os.path.join(args.output_dir, 'memory')
            plot_memory_scaling(df, memory_dir)
            create_memory_heatmap(df, memory_dir)
        
        if run_all or 'weak' in analyses:
            weak_dir = os.path.join(args.output_dir, 'weak_scaling')
            plot_weak_scaling(df, weak_dir)
            plot_weak_scaling_overview(df, weak_dir)
        
        if run_all or 'thread' in analyses:
            thread_dir = os.path.join(args.output_dir, 'thread_speedup')
            plot_thread_speedup(df, thread_dir)
            create_speedup_heatmap(df, thread_dir)
        
        if run_all or 'process' in analyses:
            process_dir = os.path.join(args.output_dir, 'thread_vs_process')
            plot_thread_vs_process(df, process_dir)
        
        if run_all or 'combined' in analyses:
            combined_dir = os.path.join(args.output_dir, 'combined_scaling')
            plot_combined_scaling(df, combined_dir)

        
        # Print summary
        end_time = time.time()
        total_time = end_time - start_time
        print("\n" + "="*50)
        print(f"Scaling analysis completed in {total_time:.2f} seconds")
        print(f"Output directory: {args.output_dir}")
        
        # Count plots
        total_plots = sum(len(files) for _, _, files in os.walk(args.output_dir) 
                         if any(file.endswith('.png') for file in files))
        print(f"Total plots generated: {total_plots}")
        
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()