# src/gpu_rf_kick.jl
"""
    gpu_rf_kick.jl - GPU implementation of RF kick

This file implements the RF cavity voltage application for particles
using CUDA acceleration.
"""

using CUDA

"""
    rf_kick_kernel!(z, ΔE, voltage, sin_ϕs, rf_factor, ϕs)

CUDA kernel for applying RF cavity voltage to particles.

# Args
- `z::CuDeviceVector{T}`: Particle longitudinal positions
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `voltage::T`: RF voltage
- `sin_ϕs::T`: Sine of synchronous phase
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
"""
function rf_kick_kernel!(z, ΔE, voltage, sin_ϕs, rf_factor, ϕs)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(z)
        ϕ_val = -z[idx] * rf_factor + ϕs
        ΔE[idx] += voltage * (sin(ϕ_val) - sin_ϕs)
    end
    return nothing
end

"""
    rf_kick_gpu!(
        gpu_data::GPUParticleData{T},
        voltage::T, sin_ϕs::T, rf_factor::T, ϕs::T,
        gpu_config::GPUConfig
    ) where T<:Float64

Apply RF kick to particles on GPU.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `voltage::T`: RF voltage
- `sin_ϕs::T`: Sine of synchronous phase
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
- `gpu_config::GPUConfig`: GPU kernel configuration
"""
function rf_kick_gpu!(
    gpu_data::GPUParticleData{T},
    voltage::T, sin_ϕs::T, rf_factor::T, ϕs::T,
    gpu_config::GPUConfig
) where T<:Float64
    n_particles = length(gpu_data.z)
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
        @cuda prefer_l1=true blocks=blocks threads=threads rf_kick_kernel!(
            gpu_data.z, gpu_data.ΔE, voltage, sin_ϕs, rf_factor, ϕs
        )
    else
        @cuda blocks=blocks threads=threads rf_kick_kernel!(
            gpu_data.z, gpu_data.ΔE, voltage, sin_ϕs, rf_factor, ϕs
        )
    end
    
    return nothing
end