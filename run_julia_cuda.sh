#!/bin/bash
# Save original LD_LIBRARY_PATH
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Remove all CUDA paths from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/opt/software-current/2023.06/x86_64/generic/software/CUDA/[^:]*:||g')

# Set CUDA.jl environment variables
export JULIA_CUDA_USE_BINARYBUILDER=true
export JULIA_CUDA_MEMORY_POOL=binned

# Run Julia with arguments
julia "$@"

# Restore original LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
