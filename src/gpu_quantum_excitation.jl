# src/gpu_quantum_excitation.jl
"""
    gpu_quantum_excitation.jl - GPU implementation of quantum excitation

This file implements the quantum excitation effects for particles
using CUDA acceleration.
"""

using CUDA

"""
    quantum_excitation_kernel!(ΔE, random_buffer, E0, radius, σ_E0)

CUDA kernel for applying quantum excitation to particles.

# Args
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `random_buffer::CuDeviceVector{T}`: Pre-generated random numbers
- `E0::T`: Reference energy
- `radius::T`: Accelerator radius
- `σ_E0::T`: Initial energy spread
"""
function quantum_excitation_kernel!(ΔE, random_buffer, E0, radius, σ_E0)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(ΔE)
        ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
        excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E0
        ΔE[idx] += excitation * random_buffer[idx]
    end
    return nothing
end

"""
    quantum_excitation_gpu!(
        gpu_data::GPUParticleData{T},
        gpu_buffers::GPUSimulationBuffers{T},
        E0::T, radius::T, σ_E0::T,
        gpu_config::GPUConfig
    ) where T<:Float64

Apply quantum excitation to particles on GPU.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `gpu_buffers::GPUSimulationBuffers{T}`: GPU simulation buffers
- `E0::T`: Reference energy
- `radius::T`: Accelerator radius
- `σ_E0::T`: Initial energy spread
- `gpu_config::GPUConfig`: GPU kernel configuration
"""
function quantum_excitation_gpu!(
    gpu_data::GPUParticleData{T},
    gpu_buffers::GPUSimulationBuffers{T},
    E0::T, radius::T, σ_E0::T,
    gpu_config::GPUConfig
) where T<:Float64
    n_particles = length(gpu_data.ΔE)
    if n_particles == 0
        return nothing
    end
    
    # Generate random numbers on GPU
    CUDA.randn!(gpu_buffers.random_buffer)
    
    # Prepare kernel launch
    threads = gpu_config.threads_per_block
    blocks = min(
        cld(n_particles, threads), 
        gpu_config.max_blocks > 0 ? gpu_config.max_blocks : typemax(Int)
    )
    
    # Launch kernel
    if gpu_config.prefer_l1_cache
        @cuda prefer_l1=true blocks=blocks threads=threads quantum_excitation_kernel!(
            gpu_data.ΔE, gpu_buffers.random_buffer, E0, radius, σ_E0
        )
    else
        @cuda blocks=blocks threads=threads quantum_excitation_kernel!(
            gpu_data.ΔE, gpu_buffers.random_buffer, E0, radius, σ_E0
        )
    end
    
    return nothing
end