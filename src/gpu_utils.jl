# src/gpu_utils.jl
"""
    gpu_utils.jl - Utility functions for GPU operations

This file provides utility functions for GPU operations such as
copying data between CPU and GPU and checking for GPU-aware MPI.
"""

using CUDA
using MPI

"""
    is_gpu_aware_mpi_available(comm::MPI.Comm)

Check if GPU-aware MPI is available.

# Args
- `comm::MPI.Comm`: MPI communicator

# Returns
- `Bool`: True if GPU-aware MPI is available
"""
function is_gpu_aware_mpi_available(comm::MPI.Comm)
    # Try a simple GPU-aware MPI operation
    try
        if CUDA.functional()
            test_buffer = CUDA.zeros(Float64, 1)
            MPI.Allreduce!(test_buffer, MPI.SUM, comm)
            return true
        end
    catch
        # If it fails, GPU-aware MPI is not available
    end
    return false
end

"""
    update_E0_gpu!(
        gpu_data::GPUParticleData{T},
        ΔE_initial_turn::Vector{T},
        E0::T,
        comm::MPI.Comm,
        use_mpi::Bool
    ) where T<:Float64

Update reference energy E0 using the average energy change on GPU.

# Args
- `gpu_data::GPUParticleData{T}`: GPU particle data
- `ΔE_initial_turn::Vector{T}`: Initial energy deviations this turn
- `E0::T`: Reference energy
- `comm::MPI.Comm`: MPI communicator
- `use_mpi::Bool`: Flag for MPI mode

# Returns
- `Tuple{T, T}`: Updated E0 and mean energy change
"""
function update_E0_gpu!(
    gpu_data::GPUParticleData{T},
    ΔE_initial_turn::Vector{T},
    E0::T,
    comm::MPI.Comm,
    use_mpi::Bool
) where T<:Float64
    n_local = length(gpu_data.ΔE)
    
    # Get current energy deviations from GPU
    ΔE_current = Array(gpu_data.ΔE)
    
    local mean_dE_this_turn::T
    
    if use_mpi
        # Calculate sum of ΔE changes on this rank
        local sum_dE_local = zero(T)
        if n_local > 0
            @simd for i in 1:n_local
                sum_dE_local += (ΔE_current[i] - ΔE_initial_turn[i])
            end
        end
        
        # Reduce sum of changes and local counts globally
        reductions = MPI.Allreduce([sum_dE_local, T(n_local)], MPI.SUM, comm)
        sum_dE_global = reductions[1]
        n_global_count = Int(round(reductions[2]))
        
        # Calculate global mean energy change for this turn
        mean_dE_this_turn = sum_dE_global / n_global_count
        
        # All ranks update E0 identically with the global mean change
        E0 = E0 + mean_dE_this_turn
        
        # Re-center local particle energies relative to the NEW E0
        if n_local > 0
            @simd for i in 1:n_local
                ΔE_current[i] -= mean_dE_this_turn
            end
            # Update GPU data
            copyto!(gpu_data.ΔE, ΔE_current)
        end
    else # Serial mode
        if n_local > 0
            # Calculate local mean energy change (which is global in serial)
            local sum_diff = zero(T)
            @simd for i in 1:n_local
                sum_diff += (ΔE_current[i] - ΔE_initial_turn[i])
            end
            mean_dE_this_turn = sum_diff / n_local
            
            # Update E0
            E0 = E0 + mean_dE_this_turn
            
            # Re-center particle energies
            @simd for i in 1:n_local
                ΔE_current[i] -= mean_dE_this_turn
            end
            # Update GPU data
            copyto!(gpu_data.ΔE, ΔE_current)
        else
            mean_dE_this_turn = zero(T)
        end
    end
    
    return E0, mean_dE_this_turn
end

"""
    compute_global_std_gpu(
        gpu_vector::CuVector{T},
        comm::MPI.Comm,
        use_mpi::Bool
    ) where T<:Float64

Calculate global standard deviation of GPU vector data.

# Args
- `gpu_vector::CuVector{T}`: GPU vector data
- `comm::MPI.Comm`: MPI communicator
- `use_mpi::Bool`: Flag for MPI mode

# Returns
- `T`: Global standard deviation
"""
function compute_global_std_gpu(
    gpu_vector::CuVector{T},
    comm::MPI.Comm,
    use_mpi::Bool
) where T<:Float64
    # Copy data to CPU for calculations
    local_data = Array(gpu_vector)
    n_local = length(local_data)
    
    # Calculate local statistics
    sum_local = zero(T)
    sum_sq_local = zero(T)
    
    if n_local > 0
        @simd for i in 1:n_local
            val = local_data[i]
            sum_local += val
            sum_sq_local += val * val
        end
    end
    
    if use_mpi
        # Reduce statistics globally
        reductions = MPI.Allreduce([sum_local, sum_sq_local, T(n_local)], MPI.SUM, comm)
        sum_global = reductions[1]
        sum_sq_global = reductions[2]
        n_global = Int(round(reductions[3]))
        
        # Calculate global standard deviation
        if n_global <= 1
            return zero(T)
        end
        
        mean_global = sum_global / n_global
        variance_global = (sum_sq_global / n_global) - (mean_global * mean_global)
        sample_variance_global = max(zero(T), variance_global * n_global / (n_global - 1))
        
        return sqrt(sample_variance_global)
    else
        # Local calculation for serial mode
        if n_local <= 1
            return zero(T)
        end
        
        mean_local = sum_local / n_local
        variance_local = (sum_sq_local / n_local) - (mean_local * mean_local)
        sample_variance_local = max(zero(T), variance_local * n_local / (n_local - 1))
        
        return sqrt(sample_variance_local)
    end
end