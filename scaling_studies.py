#!/usr/bin/env python3
"""
Benchmark Log Analysis Script

This script processes benchmark log files, extracts relevant metrics,
saves them to a CSV file, and generates scaling plots.
"""

import os
import re
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ----------------------
# Data Parsing Functions
# ----------------------

def parse_log_file(file_path):
    """
    Parse a benchmark log file and extract relevant metrics.
    
    Args:
        file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing extracted metrics
    """
    data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract parameters
    run_mode_match = re.search(r'run_mode\s*=\s*(\w+)', content)
    data['run_mode'] = run_mode_match.group(1) if run_mode_match else None
    
    turns_match = re.search(r'n_turns\s*=\s*(\d+)', content)
    data['n_turns'] = int(turns_match.group(1)) if turns_match else None
    
    particles_match = re.search(r'n_particles_global\s*=\s*(\d+)', content)
    data['n_particles_global'] = int(particles_match.group(1)) if particles_match else None
    
    threads_match = re.search(r'num_threads_per_process\s*=\s*(\d+)', content)
    data['num_threads'] = int(threads_match.group(1)) if threads_match else None
    
    mpi_size_match = re.search(r'mpi_comm_size\s*=\s*(\d+)', content)
    data['mpi_ranks'] = int(mpi_size_match.group(1)) if mpi_size_match else 1  # Default to 1 if not MPI
    
    # Extract performance metrics
    median_time_match = re.search(r'Max Median Time \(across ranks\):\s*([\d.]+)\s*(\w+)', content)
    if median_time_match:
        time_value = float(median_time_match.group(1))
        time_unit = median_time_match.group(2)
        # Convert to seconds
        if time_unit == 'ms':
            time_value /= 1000
        elif time_unit == 'μs':
            time_value /= 1000000
        elif time_unit == 'ns':
            time_value /= 1000000000
        data['median_time_seconds'] = time_value
    else:
        data['median_time_seconds'] = None
    
    # Extract memory allocation
    memory_match = re.search(r'Sum of Median Memory Allocated:\s*([\d.]+)\s*(\w+)', content)
    if memory_match:
        mem_value = float(memory_match.group(1))
        mem_unit = memory_match.group(2)
        # Convert to bytes
        if mem_unit == 'KiB':
            mem_value *= 1024
        elif mem_unit == 'MiB':
            mem_value *= 1024 * 1024
        elif mem_unit == 'GiB':
            mem_value *= 1024 * 1024 * 1024
        elif mem_unit == 'TiB':
            mem_value *= 1024 * 1024 * 1024 * 1024
        data['memory_bytes'] = mem_value
    else:
        data['memory_bytes'] = None
    
    # Extract allocations
    allocs_match = re.search(r'Sum of Median Allocations:\s*([\d.e+]+)', content)
    data['allocations'] = float(allocs_match.group(1)) if allocs_match else None
    
    # Extract FLOPs if available
    total_flops_match = re.search(r'Total aggregated FLOPs:\s*([\d.e+]+)', content)
    data['total_flops'] = float(total_flops_match.group(1)) if total_flops_match else None
    
    flops_rate_match = re.search(r'Aggregated GFLOPS rate.*:\s*([\d.]+)', content)
    data['gflops_rate'] = float(flops_rate_match.group(1)) if flops_rate_match else None
    
    flops_per_particle_turn_match = re.search(r'FLOPs per particle per turn:\s*([\d.]+)', content)
    data['flops_per_particle_turn'] = float(flops_per_particle_turn_match.group(1)) if flops_per_particle_turn_match else None
    
    # Extract the filename to check for additional info
    filename = os.path.basename(file_path)
    data['filename'] = filename
    
    # Calculate derived metrics
    if data['median_time_seconds'] is not None and data['n_turns'] is not None and data['n_particles_global'] is not None:
        data['time_per_particle_turn'] = data['median_time_seconds'] / (data['n_turns'] * data['n_particles_global'])
    else:
        data['time_per_particle_turn'] = None
        
    # Add file path for reference
    data['log_file'] = file_path
    
    return data

def process_log_files(directory):
    """
    Process all log files in the specified directory.
    
    Args:
        directory (str): Directory containing log files
        
    Returns:
        list: List of dictionaries containing extracted metrics
    """
    results = []
    
    # Find all log files
    log_files = glob.glob(os.path.join(directory, "**/*.log"), recursive=True)
    
    if not log_files:
        print(f"No log files found in {directory}")
        return results
    
    print(f"Found {len(log_files)} log files")
    
    for log_file in log_files:
        try:
            data = parse_log_file(log_file)
            results.append(data)
            print(f"Processed: {log_file}")
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return results

def save_to_csv(data, output_file):
    """
    Save the extracted data to a CSV file.
    
    Args:
        data (list): List of dictionaries containing extracted metrics
        output_file (str): Path to output CSV file
    """
    if not data:
        print("No data to save.")
        return
    
    # Get all field names
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    # Prioritize certain fields to appear first in the CSV
    priority_fields = [
        'run_mode', 'n_turns', 'n_particles_global', 'mpi_ranks', 'num_threads',
        'median_time_seconds', 'memory_bytes', 'allocations',
        'total_flops', 'gflops_rate', 'flops_per_particle_turn',
        'time_per_particle_turn'
    ]
    
    # Sort the remaining fields
    remaining_fields = sorted(fieldnames - set(priority_fields))
    
    # Final ordered field list
    final_fieldnames = [f for f in priority_fields if f in fieldnames] + remaining_fields
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {output_file}")

# ------------------------
# Strong Scaling Analysis
# ------------------------

def create_strong_scaling_plots(df, output_dir):
    """
    Create strong scaling plots (fixed problem size, varying number of processes).
    
    Args:
        df (DataFrame): Data frame with benchmark results
        output_dir (str): Directory to save plots
    """
    # Add a column for total cores (mpi_ranks * num_threads)
    if 'num_threads' in df.columns:
        df['total_cores'] = df['mpi_ranks'] * df['num_threads']
    else:
        df['total_cores'] = df['mpi_ranks']  # Fallback if threads info not available
    
    # Group by problem size (n_turns and n_particles_global)
    problem_sizes = df.groupby(['n_turns', 'n_particles_global'])
    
    # Create subdirectory for strong scaling plots
    strong_dir = os.path.join(output_dir, 'strong_scaling')
    os.makedirs(strong_dir, exist_ok=True)
    
    # Color map for different thread counts
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for (n_turns, n_particles), group in problem_sizes:
        if len(group) < 2:
            print(f"Skipping problem size (turns={n_turns}, particles={n_particles}) - not enough data points")
            continue
        
        # Create a subdirectory for this problem size
        prob_dir = os.path.join(strong_dir, f'turns{n_turns}_particles{n_particles}')
        os.makedirs(prob_dir, exist_ok=True)
        
        # For each problem size, group by thread count
        thread_groups = group.groupby('num_threads')
        
        # Skip if not enough thread groups
        if len(thread_groups) < 1:
            print(f"Skipping problem size (turns={n_turns}, particles={n_particles}) - no thread groups")
            continue
        
        # Plot 1: Execution time vs. number of ranks
        plt.figure(figsize=(10, 6))
        
        # Reference for ideal scaling - use minimum ranks as base
        min_ranks = group['mpi_ranks'].min()
        
        # Iterate over thread groups and plot each with a different color/marker
        for i, (thread_count, thread_group) in enumerate(thread_groups):
            if len(thread_group) < 2:
                # Just plot the point without a line if only one data point
                plt.plot(thread_group['mpi_ranks'], thread_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
            else:
                # Sort by number of MPI ranks
                thread_group = thread_group.sort_values('mpi_ranks')
                plt.plot(thread_group['mpi_ranks'], thread_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
        
        # Add ideal scaling line
        # Find the point with minimum ranks
        min_rank_row = group[group['mpi_ranks'] == min_ranks].iloc[0]
        base_time = min_rank_row['median_time_seconds']
        base_ranks = min_rank_row['mpi_ranks']
        
        # Generate x points for ideal line
        x_ideal = np.array(sorted(group['mpi_ranks'].unique()))
        # Calculate ideal scaling
        y_ideal = base_time * base_ranks / x_ideal
        
        plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling')
        
        plt.title(f'Strong Scaling: Execution Time\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Time (seconds)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'execution_time.png'))
        plt.close()
        
        # Plot 2: Speedup vs. number of ranks
        plt.figure(figsize=(10, 6))
        
        # Find the point with minimum ranks for reference
        min_rank_row = group[group['mpi_ranks'] == min_ranks].iloc[0]
        base_time = min_rank_row['median_time_seconds']
        base_ranks = min_rank_row['mpi_ranks']
        
        # Plot speedup for each thread group
        for i, (thread_count, thread_group) in enumerate(thread_groups):
            if len(thread_group) < 2:
                # Just plot the point without a line
                speedup = [base_time / t for t in thread_group['median_time_seconds']]
                plt.plot(thread_group['mpi_ranks'], speedup, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
            else:
                # Sort by number of MPI ranks
                thread_group = thread_group.sort_values('mpi_ranks')
                speedup = [base_time / t for t in thread_group['median_time_seconds']]
                plt.plot(thread_group['mpi_ranks'], speedup, 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
        
        # Add ideal speedup line
        x_ideal = np.array(sorted(group['mpi_ranks'].unique()))
        y_ideal = x_ideal / base_ranks
        plt.plot(x_ideal, y_ideal, 'k--', label='Ideal scaling')
        
        plt.title(f'Strong Scaling: Speedup\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Speedup')
        plt.xscale('log', base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'speedup.png'))
        plt.close()
        
        # Plot 3: Efficiency vs. number of ranks
        plt.figure(figsize=(10, 6))
        
        # Find the point with minimum ranks for reference
        min_rank_row = group[group['mpi_ranks'] == min_ranks].iloc[0]
        base_time = min_rank_row['median_time_seconds']
        base_ranks = min_rank_row['mpi_ranks']
        
        for i, (thread_count, thread_group) in enumerate(thread_groups):
            if len(thread_group) < 2:
                # Just plot the point without a line
                speedup = [base_time / t for t in thread_group['median_time_seconds']]
                efficiency = [s / (r / base_ranks) for s, r in zip(speedup, thread_group['mpi_ranks'])]
                plt.plot(thread_group['mpi_ranks'], efficiency, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
            else:
                # Sort by number of MPI ranks
                thread_group = thread_group.sort_values('mpi_ranks')
                speedup = [base_time / t for t in thread_group['median_time_seconds']]
                efficiency = [s / (r / base_ranks) for s, r in zip(speedup, thread_group['mpi_ranks'])]
                plt.plot(thread_group['mpi_ranks'], efficiency, 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{thread_count} threads')
        
        plt.title(f'Strong Scaling: Efficiency\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Efficiency')
        plt.xscale('log', base=2)
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'efficiency.png'))
        plt.close()
        
        # Plot 4: Memory usage vs. number of ranks
        if 'memory_bytes' in group.columns and group['memory_bytes'].notna().all():
            plt.figure(figsize=(10, 6))
            
            for i, (thread_count, thread_group) in enumerate(thread_groups):
                if len(thread_group) < 2:
                    # Just plot the point without a line
                    plt.plot(thread_group['mpi_ranks'], thread_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
                else:
                    # Sort by number of MPI ranks
                    thread_group = thread_group.sort_values('mpi_ranks')
                    plt.plot(thread_group['mpi_ranks'], thread_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
            
            plt.title(f'Strong Scaling: Memory Usage\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('Memory (GiB)')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'memory_usage.png'))
            plt.close()
        
        # Plot 5: GFLOPS rate vs. number of ranks (if available)
        if 'gflops_rate' in group.columns and group['gflops_rate'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (thread_count, thread_group) in enumerate(thread_groups):
                if len(thread_group) < 2:
                    # Just plot the point without a line
                    plt.plot(thread_group['mpi_ranks'], thread_group['gflops_rate'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
                else:
                    # Sort by number of MPI ranks
                    thread_group = thread_group.sort_values('mpi_ranks')
                    plt.plot(thread_group['mpi_ranks'], thread_group['gflops_rate'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
            
            plt.title(f'Strong Scaling: GFLOPS Rate\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('GFLOPS')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'gflops_rate.png'))
            plt.close()
        
        # Plot 6: Allocations vs. number of ranks (if available)
        if 'allocations' in group.columns and group['allocations'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (thread_count, thread_group) in enumerate(thread_groups):
                if len(thread_group) < 2:
                    # Just plot the point without a line
                    plt.plot(thread_group['mpi_ranks'], thread_group['allocations'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
                else:
                    # Sort by number of MPI ranks
                    thread_group = thread_group.sort_values('mpi_ranks')
                    plt.plot(thread_group['mpi_ranks'], thread_group['allocations'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{thread_count} threads')
            
            plt.title(f'Strong Scaling: Memory Allocations\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('Number of Allocations')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'allocations.png'))
            plt.close()

# -----------------------
# Weak Scaling Analysis
# -----------------------

def create_weak_scaling_plots(df, output_dir):
    """
    Create weak scaling plots with:
    - X-axis: Number of ranks
    - Y-axis: Performance metrics (GFLOPS, time, etc.)
    - Different lines: Different TOTAL particle counts (not per rank)
    - Separate plots for each thread count
    
    Args:
        df (DataFrame): Data frame with benchmark results
        output_dir (str): Directory to save plots
    """
    # Calculate particles per rank
    df['particles_per_rank'] = df['n_particles_global'] / df['mpi_ranks']
    
    # Add a column for total cores (mpi_ranks * num_threads)
    if 'num_threads' in df.columns:
        df['total_cores'] = df['mpi_ranks'] * df['num_threads']
    else:
        df['total_cores'] = df['mpi_ranks']  # Fallback if threads info not available
    
    # Create subdirectory for weak scaling plots
    weak_dir = os.path.join(output_dir, 'weak_scaling')
    os.makedirs(weak_dir, exist_ok=True)
    
    # Color map for different particle counts
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Group by number of turns and thread count
    thread_groups = df.groupby(['n_turns', 'num_threads'])
    
    for (n_turns, n_threads), thread_group in thread_groups:
        # Skip if not enough data points
        if len(thread_group) < 2:
            print(f"Skipping weak scaling for turns={n_turns}, threads={n_threads} - not enough data points")
            continue
        
        # Create a subdirectory for this thread configuration
        thread_dir = os.path.join(weak_dir, f'turns{n_turns}_threads{n_threads}')
        os.makedirs(thread_dir, exist_ok=True)
        
        # Group by total particle count
        particles_groups = thread_group.groupby('n_particles_global')
        
        # Skip if not enough particle groups
        if len(particles_groups) < 2:
            print(f"Skipping weak scaling for turns={n_turns}, threads={n_threads} - not enough particle counts")
            continue
        
        # Plot 1: Execution time vs. number of ranks
        plt.figure(figsize=(10, 6))
        
        for i, (total_particles, particle_group) in enumerate(particles_groups):
            if len(particle_group) < 2:
                # Just plot the point without a line
                plt.plot(particle_group['mpi_ranks'], particle_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{total_particles:.1e} total particles')
            else:
                # Sort by number of ranks
                particle_group = particle_group.sort_values('mpi_ranks')
                plt.plot(particle_group['mpi_ranks'], particle_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{total_particles:.1e} total particles')
        
        plt.title(f'Weak Scaling: Execution Time\n(Turns={n_turns}, Threads={n_threads})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Time (seconds)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(thread_dir, 'execution_time_vs_ranks.png'))
        plt.close()
        
        # Plot 2: Memory usage vs. number of ranks
        if 'memory_bytes' in thread_group.columns and thread_group['memory_bytes'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (total_particles, particle_group) in enumerate(particles_groups):
                if len(particle_group) < 2:
                    # Just plot the point without a line
                    plt.plot(particle_group['mpi_ranks'], particle_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
                else:
                    # Sort by number of ranks
                    particle_group = particle_group.sort_values('mpi_ranks')
                    plt.plot(particle_group['mpi_ranks'], particle_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
            
            plt.title(f'Weak Scaling: Memory Usage\n(Turns={n_turns}, Threads={n_threads})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('Memory (GiB)')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(thread_dir, 'memory_vs_ranks.png'))
            plt.close()
        
        # Plot 3: GFLOPS rate vs. number of ranks (if available)
        if 'gflops_rate' in thread_group.columns and thread_group['gflops_rate'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (total_particles, particle_group) in enumerate(particles_groups):
                if len(particle_group) < 2:
                    # Just plot the point without a line
                    plt.plot(particle_group['mpi_ranks'], particle_group['gflops_rate'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
                else:
                    # Sort by number of ranks
                    particle_group = particle_group.sort_values('mpi_ranks')
                    plt.plot(particle_group['mpi_ranks'], particle_group['gflops_rate'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
            
            plt.title(f'Weak Scaling: GFLOPS Rate\n(Turns={n_turns}, Threads={n_threads})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('GFLOPS')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(thread_dir, 'gflops_rate_vs_ranks.png'))
            plt.close()
            
            # Plot 3B: GFLOPS per rank vs. number of ranks
            plt.figure(figsize=(10, 6))
            
            for i, (total_particles, particle_group) in enumerate(particles_groups):
                if len(particle_group) < 2:
                    # Just plot the point without a line
                    gflops_per_rank = particle_group['gflops_rate'] / particle_group['mpi_ranks']
                    plt.plot(particle_group['mpi_ranks'], gflops_per_rank, 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
                else:
                    # Sort by number of ranks
                    particle_group = particle_group.sort_values('mpi_ranks')
                    gflops_per_rank = particle_group['gflops_rate'] / particle_group['mpi_ranks']
                    plt.plot(particle_group['mpi_ranks'], gflops_per_rank, 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
            
            plt.title(f'Weak Scaling: GFLOPS per Rank\n(Turns={n_turns}, Threads={n_threads})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('GFLOPS per Rank')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(thread_dir, 'gflops_per_rank_vs_ranks.png'))
            plt.close()
        
        # Plot 4: Allocations vs. number of ranks (if available)
        if 'allocations' in thread_group.columns and thread_group['allocations'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (total_particles, particle_group) in enumerate(particles_groups):
                if len(particle_group) < 2:
                    # Just plot the point without a line
                    plt.plot(particle_group['mpi_ranks'], particle_group['allocations'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
                else:
                    # Sort by number of ranks
                    particle_group = particle_group.sort_values('mpi_ranks')
                    plt.plot(particle_group['mpi_ranks'], particle_group['allocations'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{total_particles:.1e} total particles')
            
            plt.title(f'Weak Scaling: Memory Allocations\n(Turns={n_turns}, Threads={n_threads})')
            plt.xlabel('MPI Ranks')
            plt.ylabel('Number of Allocations')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(thread_dir, 'allocations_vs_ranks.png'))
            plt.close()
            
        # Plot 5: Time per particle vs. number of ranks
        plt.figure(figsize=(10, 6))
        
        for i, (total_particles, particle_group) in enumerate(particles_groups):
            if len(particle_group) < 2:
                # Just plot the point without a line
                time_per_particle = particle_group['median_time_seconds'] / total_particles
                plt.plot(particle_group['mpi_ranks'], time_per_particle, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{total_particles:.1e} total particles')
            else:
                # Sort by number of ranks
                particle_group = particle_group.sort_values('mpi_ranks')
                time_per_particle = particle_group['median_time_seconds'] / total_particles
                plt.plot(particle_group['mpi_ranks'], time_per_particle, 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{total_particles:.1e} total particles')
        
        plt.title(f'Weak Scaling: Time per Particle\n(Turns={n_turns}, Threads={n_threads})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Time per Particle (s)')
        plt.xscale('log', base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(thread_dir, 'time_per_particle_vs_ranks.png'))
        plt.close()
        
        # Plot 6: Efficiency vs. number of ranks (for weak scaling)
        plt.figure(figsize=(10, 6))
        
        for i, (total_particles, particle_group) in enumerate(particles_groups):
            if len(particle_group) < 2:
                # Not enough points for efficiency calculation
                continue
            
            # Sort by number of ranks
            particle_group = particle_group.sort_values('mpi_ranks')
            
            # Calculate efficiency: T_1 / T_n (for weak scaling, should be close to 1.0 for good scaling)
            min_ranks = particle_group['mpi_ranks'].min()
            base_time = particle_group[particle_group['mpi_ranks'] == min_ranks]['median_time_seconds'].values[0]
            
            # Calculate efficiency relative to the minimum rank count
            relative_efficiency = [base_time / t for t in particle_group['median_time_seconds']]
            
            plt.plot(particle_group['mpi_ranks'], relative_efficiency, 
                     marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                     label=f'{total_particles:.1e} total particles')
        
        plt.title(f'Weak Scaling: Efficiency\n(Turns={n_turns}, Threads={n_threads})')
        plt.xlabel('MPI Ranks')
        plt.ylabel('Efficiency (T_1 / T_n)')
        plt.xscale('log', base=2)
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(thread_dir, 'efficiency_vs_ranks.png'))
        plt.close()

# ----------------------
# Thread Scaling Analysis
# ----------------------

def create_threads_scaling_plots(df, output_dir):
    """
    Create scaling plots with threads on the x-axis and different ranks as colors.
    
    Args:
        df (DataFrame): Data frame with benchmark results
        output_dir (str): Directory to save plots
    """
    # Create subdirectory for thread scaling plots
    threads_dir = os.path.join(output_dir, 'threads_scaling')
    os.makedirs(threads_dir, exist_ok=True)
    
    # Color map for different rank counts
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Group by problem size (n_turns and n_particles_global)
    problem_sizes = df.groupby(['n_turns', 'n_particles_global'])
    
    for (n_turns, n_particles), group in problem_sizes:
        if len(group) < 2:
            print(f"Skipping threads scaling for (turns={n_turns}, particles={n_particles}) - not enough data points")
            continue
        
        # Skip if there's only one thread count
        if len(group['num_threads'].unique()) < 2:
            print(f"Skipping threads scaling for (turns={n_turns}, particles={n_particles}) - only one thread count")
            continue
        
        # Create a subdirectory for this problem size
        prob_dir = os.path.join(threads_dir, f'turns{n_turns}_particles{n_particles}')
        os.makedirs(prob_dir, exist_ok=True)
        
        # Group by MPI ranks
        rank_groups = group.groupby('mpi_ranks')
        
        # Plot 1: Execution time vs. number of threads
        plt.figure(figsize=(10, 6))
        
        for i, (rank_count, rank_group) in enumerate(rank_groups):
            if len(rank_group) < 2:
                # Just plot the point without a line
                plt.plot(rank_group['num_threads'], rank_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                         label=f'{rank_count} ranks')
            else:
                # Sort by number of threads
                rank_group = rank_group.sort_values('num_threads')
                plt.plot(rank_group['num_threads'], rank_group['median_time_seconds'], 
                         marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                         label=f'{rank_count} ranks')
        
        plt.title(f'Thread Scaling: Execution Time\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('Number of Threads')
        plt.ylabel('Time (seconds)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'execution_time_vs_threads.png'))
        plt.close()
        
        # Plot 2: Speedup vs. number of threads (relative to single thread for each rank count)
        plt.figure(figsize=(10, 6))
        
        for i, (rank_count, rank_group) in enumerate(rank_groups):
            if len(rank_group) < 2:
                continue  # Skip if only one thread count
            
            # Sort by number of threads
            rank_group = rank_group.sort_values('num_threads')
            
            # Get the single-thread run (or the minimum thread count available)
            min_threads = rank_group['num_threads'].min()
            base_time = rank_group[rank_group['num_threads'] == min_threads]['median_time_seconds'].values[0]
            
            # Calculate speedup
            speedup = [base_time / t for t in rank_group['median_time_seconds']]
            
            plt.plot(rank_group['num_threads'], speedup, 
                     marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                     label=f'{rank_count} ranks')
            
            # Add ideal scaling line specific to this rank count
            min_threads_for_rank = rank_group['num_threads'].min()
            max_threads_for_rank = rank_group['num_threads'].max()
            if min_threads_for_rank != max_threads_for_rank:
                ideal_x = np.array([min_threads_for_rank, max_threads_for_rank])
                ideal_y = ideal_x / min_threads_for_rank
                plt.plot(ideal_x, ideal_y, '--', color=colors[i % len(colors)], alpha=0.5)
        
        # Add a global ideal scaling reference line
        min_threads_global = group['num_threads'].min()
        max_threads_global = group['num_threads'].max()
        if min_threads_global != max_threads_global:
            ideal_x_global = np.array([min_threads_global, max_threads_global])
            ideal_y_global = ideal_x_global / min_threads_global
            plt.plot(ideal_x_global, ideal_y_global, 'k--', label='Ideal scaling')
        
        plt.title(f'Thread Scaling: Speedup\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup')
        plt.xscale('log', base=2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'speedup_vs_threads.png'))
        plt.close()
        
        # Plot 3: Efficiency vs. number of threads
        plt.figure(figsize=(10, 6))
        
        for i, (rank_count, rank_group) in enumerate(rank_groups):
            if len(rank_group) < 2:
                continue  # Skip if only one thread count
            
            # Sort by number of threads
            rank_group = rank_group.sort_values('num_threads')
            
            # Get the single-thread run (or the minimum thread count available)
            min_threads = rank_group['num_threads'].min()
            base_time = rank_group[rank_group['num_threads'] == min_threads]['median_time_seconds'].values[0]
            
            # Calculate speedup and efficiency
            speedup = [base_time / t for t in rank_group['median_time_seconds']]
            efficiency = [s / (t / min_threads) for s, t in zip(speedup, rank_group['num_threads'])]
            
            plt.plot(rank_group['num_threads'], efficiency, 
                     marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                     label=f'{rank_count} ranks')
        
        plt.title(f'Thread Scaling: Efficiency\n(Turns={n_turns}, Particles={n_particles})')
        plt.xlabel('Number of Threads')
        plt.ylabel('Efficiency')
        plt.xscale('log', base=2)
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prob_dir, 'efficiency_vs_threads.png'))
        plt.close()
        
        # Plot 4: Memory usage vs. number of threads
        if 'memory_bytes' in group.columns and group['memory_bytes'].notna().all():
            plt.figure(figsize=(10, 6))
            
            for i, (rank_count, rank_group) in enumerate(rank_groups):
                if len(rank_group) < 2:
                    # Just plot the point without a line
                    plt.plot(rank_group['num_threads'], rank_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
                else:
                    # Sort by number of threads
                    rank_group = rank_group.sort_values('num_threads')
                    plt.plot(rank_group['num_threads'], rank_group['memory_bytes'] / (1024**3), 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
            
            plt.title(f'Thread Scaling: Memory Usage\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('Number of Threads')
            plt.ylabel('Memory (GiB)')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'memory_vs_threads.png'))
            plt.close()
        
        # Plot 5: GFLOPS rate vs. number of threads (if available)
        if 'gflops_rate' in group.columns and group['gflops_rate'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (rank_count, rank_group) in enumerate(rank_groups):
                if len(rank_group) < 2:
                    # Just plot the point without a line
                    plt.plot(rank_group['num_threads'], rank_group['gflops_rate'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
                else:
                    # Sort by number of threads
                    rank_group = rank_group.sort_values('num_threads')
                    plt.plot(rank_group['num_threads'], rank_group['gflops_rate'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
            
            plt.title(f'Thread Scaling: GFLOPS Rate\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('Number of Threads')
            plt.ylabel('GFLOPS')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'gflops_rate_vs_threads.png'))
            plt.close()
            
            # Plot 5B: GFLOPS per thread vs. number of threads
            plt.figure(figsize=(10, 6))
            
            for i, (rank_count, rank_group) in enumerate(rank_groups):
                if len(rank_group) < 2:
                    # Just plot the point without a line
                    gflops_per_thread = rank_group['gflops_rate'] / (rank_group['num_threads'] * rank_count)
                    plt.plot(rank_group['num_threads'], gflops_per_thread, 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
                else:
                    # Sort by number of threads
                    rank_group = rank_group.sort_values('num_threads')
                    gflops_per_thread = rank_group['gflops_rate'] / (rank_group['num_threads'] * rank_count)
                    plt.plot(rank_group['num_threads'], gflops_per_thread, 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
            
            plt.title(f'Thread Scaling: GFLOPS per Thread\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('Number of Threads')
            plt.ylabel('GFLOPS per Thread')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'gflops_per_thread_vs_threads.png'))
            plt.close()
        
        # Plot 6: Allocations vs. number of threads (if available)
        if 'allocations' in group.columns and group['allocations'].notna().any():
            plt.figure(figsize=(10, 6))
            
            for i, (rank_count, rank_group) in enumerate(rank_groups):
                if len(rank_group) < 2:
                    # Just plot the point without a line
                    plt.plot(rank_group['num_threads'], rank_group['allocations'], 
                             marker=markers[i % len(markers)], color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
                else:
                    # Sort by number of threads
                    rank_group = rank_group.sort_values('num_threads')
                    plt.plot(rank_group['num_threads'], rank_group['allocations'], 
                             marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], 
                             label=f'{rank_count} ranks')
            
            plt.title(f'Thread Scaling: Memory Allocations\n(Turns={n_turns}, Particles={n_particles})')
            plt.xlabel('Number of Threads')
            plt.ylabel('Number of Allocations')
            plt.xscale('log', base=2)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(prob_dir, 'allocations_vs_threads.png'))
            plt.close()

# ----------------------
# Main Function
# ----------------------

def create_scaling_plots(csv_file, output_dir):
    """
    Create scaling plots from the CSV data.
    
    Args:
        csv_file (str): Path to CSV file
        output_dir (str): Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Check if we have enough data
    if len(df) < 2:
        print("Not enough data for scaling plots.")
        return
    
    # Set style
    plt.style.use('ggplot')
    
    # Create strong scaling plots (where n_turns and n_particles_global are constant)
    create_strong_scaling_plots(df, output_dir)
    
    # Create weak scaling plots (ranks on x-axis, particles per rank as different lines)
    create_weak_scaling_plots(df, output_dir)
    
    # Create plots with threads on x-axis and ranks as different colors
    create_threads_scaling_plots(df, output_dir)

def main():
    """
    Main function - parse arguments and run analysis
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process benchmark log files and generate scaling plots.')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--output-csv', type=str, default='benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--plot-dir', type=str, default='scaling_plots', help='Output directory for plots')
    args = parser.parse_args()
    
    # Process log files
    print(f"Processing log files in: {args.log_dir}")
    results = process_log_files(args.log_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Save results to CSV
    save_to_csv(results, args.output_csv)
    
    # Create scaling plots
    print(f"Creating scaling plots in: {args.plot_dir}")
    create_scaling_plots(args.output_csv, args.plot_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()