# src/gpu_data_transfer.jl
"""
    gpu_data_transfer.jl - Functions for transferring data between CPU and GPU

This file provides utilities for moving data between CPU and GPU memory
efficiently using pinned memory when available.
"""

using CUDA
using StructArrays

"""
    transfer_particles_to_gpu!(
        gpu_data::GPUParticleData{T},
        cpu_particles::StructArray{Particle{T}},
        pinned_z::Vector{T}=nothing,
        pinned_ΔE::Vector{T}=nothing
    ) where T<:Float64

Transfer particle data from CPU to GPU, optionally using pinned memory.

# Args
- `gpu_data::GPUParticleData{T}`: Destination GPU data structure
- `cpu_particles::StructArray{Particle{T}}`: Source CPU particle data
- `pinned_z::Vector{T}`: Optional pinned memory buffer for z coordinates
- `pinned_ΔE::Vector{T}`: Optional pinned memory buffer for ΔE values
"""
function transfer_particles_to_gpu!(
    gpu_data::GPUParticleData{T},
    cpu_particles::StructArray{Particle{T}},
    pinned_z::Vector{T}=nothing,
    pinned_ΔE::Vector{T}=nothing
) where T<:Float64
    n_particles = length(cpu_particles)
    if n_particles == 0
        return nothing
    end
    
    if pinned_z !== nothing && pinned_ΔE !== nothing && 
       length(pinned_z) >= n_particles && length(pinned_ΔE) >= n_particles
        # Use pinned memory path
        copyto!(pinned_z, 1, cpu_particles.coordinates.z, 1, n_particles)
        copyto!(pinned_ΔE, 1, cpu_particles.coordinates.ΔE, 1, n_particles)
        
        # Copy from pinned memory to GPU
        copyto!(gpu_data.z, 1, pinned_z, 1, n_particles)
        copyto!(gpu_data.ΔE, 1, pinned_ΔE, 1, n_particles)
    else
        # Direct path
        copyto!(gpu_data.z, cpu_particles.coordinates.z)
        copyto!(gpu_data.ΔE, cpu_particles.coordinates.ΔE)
    end
    return nothing
end

"""
    transfer_particles_to_cpu!(
        cpu_particles::StructArray{Particle{T}},
        gpu_data::GPUParticleData{T},
        pinned_z::Vector{T}=nothing,
        pinned_ΔE::Vector{T}=nothing
    ) where T<:Float64

Transfer particle data from GPU to CPU, optionally using pinned memory.

# Args
- `cpu_particles::StructArray{Particle{T}}`: Destination CPU particle data
- `gpu_data::GPUParticleData{T}`: Source GPU data structure
- `pinned_z::Vector{T}`: Optional pinned memory buffer for z coordinates
- `pinned_ΔE::Vector{T}`: Optional pinned memory buffer for ΔE values
"""
function transfer_particles_to_cpu!(
    cpu_particles::StructArray{Particle{T}},
    gpu_data::GPUParticleData{T},
    pinned_z::Vector{T}=nothing,
    pinned_ΔE::Vector{T}=nothing
) where T<:Float64
    n_particles = length(cpu_particles)
    if n_particles == 0
        return nothing
    end
    
    if pinned_z !== nothing && pinned_ΔE !== nothing && 
       length(pinned_z) >= n_particles && length(pinned_ΔE) >= n_particles
        # Use pinned memory path
        copyto!(pinned_z, 1, gpu_data.z, 1, n_particles)
        copyto!(pinned_ΔE, 1, gpu_data.ΔE, 1, n_particles)
        
        # Copy from pinned memory to CPU
        copyto!(cpu_particles.coordinates.z, 1, pinned_z, 1, n_particles)
        copyto!(cpu_particles.coordinates.ΔE, 1, pinned_ΔE, 1, n_particles)
    else
        # Direct path
        copyto!(cpu_particles.coordinates.z, gpu_data.z)
        copyto!(cpu_particles.coordinates.ΔE, gpu_data.ΔE)
    end
    return nothing
end

"""
    transfer_bin_edges_to_gpu!(
        gpu_buffers::GPUSimulationBuffers{T},
        bin_edges::AbstractRange{T}
    ) where T<:Float64

Transfer bin edges to GPU for histogram calculations.

# Args
- `gpu_buffers::GPUSimulationBuffers{T}`: GPU buffers structure
- `bin_edges::AbstractRange{T}`: Bin edges in CPU memory
"""
function transfer_bin_edges_to_gpu!(
    gpu_buffers::GPUSimulationBuffers{T},
    bin_edges::AbstractRange{T}
) where T<:Float64
    nbins = length(bin_edges) - 1
    # Copy bin edges to GPU
    copyto!(gpu_buffers.d_bin_edges, collect(bin_edges))
    
    # Calculate bin centers on GPU
    bin_centers_kernel = @cuda launch=false bin_centers_calc_kernel!(
        gpu_buffers.d_bin_centers, gpu_buffers.d_bin_edges, nbins)
    config = launch_config(bin_centers_kernel.fun, nbins)
    bin_centers_kernel(
        gpu_buffers.d_bin_centers, gpu_buffers.d_bin_edges, nbins;
        threads=config.threads, blocks=config.blocks
    )
    return nothing
end

"""
    bin_centers_calc_kernel!(centers, edges, nbins)

CUDA kernel to calculate bin centers from bin edges.

# Args
- `centers::CuDeviceVector{T}`: Output bin centers
- `edges::CuDeviceVector{T}`: Input bin edges
- `nbins::Int`: Number of bins
"""
function bin_centers_calc_kernel!(centers, edges, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nbins
        centers[idx] = (edges[idx] + edges[idx+1]) / 2
    end
    return nothing
end

"""
    launch_config(kernel_fun, n_items; threads_per_block=256)

Calculate a suitable launch configuration for a CUDA kernel.

# Args
- `kernel_fun`: CUDA kernel function
- `n_items::Int`: Number of items to process
- `threads_per_block::Int=256`: Number of threads per block

# Returns
- `NamedTuple{(:threads, :blocks), Tuple{Int, Int}}`: Launch configuration
"""
function launch_config(kernel_fun, n_items; threads_per_block=256)
    # Calculate optimal number of threads and blocks
    max_threads = CUDA.attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads = min(threads_per_block, max_threads, n_items)
    blocks = cld(n_items, threads)
    
    return (threads=threads, blocks=blocks)
end