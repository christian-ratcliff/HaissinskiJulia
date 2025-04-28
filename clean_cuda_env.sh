#!/bin/bash

# Unload all modules that might interfere
module purge
module load Julia/1.9.3-linux-x86_64
module load likwid
module load OpenMPI

# Save original LD_LIBRARY_PATH
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Remove ALL CUDA-related paths from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "CUDA" | tr '\n' ':')

# Simplified environment settings
export JULIA_CUDA_MEMORY_POOL=none  # Most compatible memory pool setting

# Run Julia with arguments
julia --project=. "$@"

# Restore original environment
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
