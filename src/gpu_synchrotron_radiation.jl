# src/gpu_synchrotron_radiation.jl
"""
    gpu_synchrotron_radiation.jl - GPU implementation of synchrotron radiation effects

This file implements the synchrotron radiation damping for particles
using CUDA acceleration.
"""

using CUDA

"""
    synchrotron_radiation_kernel!(ΔE, E0, radius)

CUDA kernel for applying synchrotron radiation damping to particles.

# Args
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `E0::T`: Reference energy
- `radius::T`: Accelerator radius
"""
function synchrotron_radiation_kernel!(ΔE, E0, radius)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(ΔE)
        ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
        damping_factor = 1 - ∂U_∂E
        ΔE[idx] *= damping_factor
    end
    return nothing
end

"""
    synchrotron_radiation_gpu!(
        gpu_data::GPUParticleData{T},
        E0::T, radius::T,
        gpu_config::GPUConfig
    ) where T<:Float64

Apply synchrotron radiation damping to particles on GPU.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `E0::T`: Reference energy
- `radius::T`: Accelerator radius
- `gpu_config::GPUConfig`: GPU kernel configuration
"""
function synchrotron_radiation_gpu!(
    gpu_data::GPUParticleData{T},
    E0::T, radius::T,
    gpu_config::GPUConfig
) where T<:Float64
    n_particles = length(gpu_data.ΔE)
    if n_particles == 0
        return nothing
    end
    
    # Prepare kernel launch
    threads = gpu_config.threads_per_block
    blocks = min(
        cld(n_particles, threads), 
        gpu_config.max_blocks > 0 ? gpu_config.max_blocks : typemax(Int)
    )
    
    # Launch kernel
    if gpu_config.prefer_l1_cache
        @cuda prefer_l1=true blocks=blocks threads=threads synchrotron_radiation_kernel!(
            gpu_data.ΔE, E0, radius
        )
    else
        @cuda blocks=blocks threads=threads synchrotron_radiation_kernel!(
            gpu_data.ΔE, E0, radius
        )
    end
    
    return nothing
end