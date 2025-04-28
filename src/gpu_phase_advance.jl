# src/gpu_phase_advance.jl
"""
    gpu_phase_advance.jl - GPU implementation of phase advance

This file implements the phase advance for particles
using CUDA acceleration.
"""

using CUDA

"""
    phase_advance_kernel!(z, ΔE, η0, harmonic, β0, E0, rf_factor, ϕs)

CUDA kernel for applying phase advance to particles with constant slip factor.

# Args
- `z::CuDeviceVector{T}`: Particle longitudinal positions
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `η0::T`: Slip factor
- `harmonic::Int`: Harmonic number
- `β0::T`: Relativistic beta
- `E0::T`: Reference energy
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
"""
function phase_advance_kernel!(z, ΔE, η0, harmonic, β0, E0, rf_factor, ϕs)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(z)
        coeff = (2π * harmonic * η0 / (β0 * β0 * E0))
        ϕ_i = -(z[idx] * rf_factor - ϕs)  # z_to_ϕ inline
        ϕ_i += coeff * ΔE[idx]            # Apply energy-dependent advance
        z[idx] = (-ϕ_i + ϕs) / rf_factor  # ϕ_to_z inline
    end
    return nothing
end

"""
    apply_phase_advance_gpu!(
        gpu_data::GPUParticleData{T},
        η0::T, harmonic::Int, β0::T, E0::T, rf_factor::T, ϕs::T,
        gpu_config::GPUConfig
    ) where T<:Float64

Apply phase advance to particles on GPU with constant slip factor.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `η0::T`: Slip factor
- `harmonic::Int`: Harmonic number
- `β0::T`: Relativistic beta
- `E0::T`: Reference energy
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
- `gpu_config::GPUConfig`: GPU kernel configuration
"""
function apply_phase_advance_gpu!(
    gpu_data::GPUParticleData{T},
    η0::T, harmonic::Int, β0::T, E0::T, rf_factor::T, ϕs::T,
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
        @cuda prefer_l1=true blocks=blocks threads=threads phase_advance_kernel!(
            gpu_data.z, gpu_data.ΔE, η0, harmonic, β0, E0, rf_factor, ϕs
        )
    else
        @cuda blocks=blocks threads=threads phase_advance_kernel!(
            gpu_data.z, gpu_data.ΔE, η0, harmonic, β0, E0, rf_factor, ϕs
        )
    end
    
    return nothing
end

"""
    phase_advance_dynamic_kernel!(z, ΔE, γ0, mass, α_c, harmonic, β0, E0, rf_factor, ϕs)

CUDA kernel for applying phase advance to particles with energy-dependent slip factor.

# Args
- `z::CuDeviceVector{T}`: Particle longitudinal positions
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `γ0::T`: Relativistic gamma
- `mass::T`: Particle mass
- `α_c::T`: Momentum compaction factor
- `harmonic::Int`: Harmonic number
- `β0::T`: Relativistic beta
- `E0::T`: Reference energy
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
"""
function phase_advance_dynamic_kernel!(z, ΔE, γ0, mass, α_c, harmonic, β0, E0, rf_factor, ϕs)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(z)
        Δγ_i = ΔE[idx] / mass
        # Avoid division by zero
        γ_particle = max(γ0 + Δγ_i, 1.0e-6)
        
        # Energy-dependent slip factor
        η_i = α_c - 1.0 / (γ_particle * γ_particle)
        
        coeff_i = (2π * harmonic * η_i / (β0 * β0 * E0))
        
        # Inline phase advance calculation
        ϕ_i = -(z[idx] * rf_factor - ϕs)  # z_to_ϕ
        ϕ_i += coeff_i * ΔE[idx]          # Apply energy-dependent advance
        z[idx] = (-ϕ_i + ϕs) / rf_factor  # ϕ_to_z
    end
    return nothing
end

"""
    apply_phase_advance_dynamic_gpu!(
        gpu_data::GPUParticleData{T},
        γ0::T, mass::T, α_c::T, 
        harmonic::Int, β0::T, E0::T, rf_factor::T, ϕs::T,
        gpu_config::GPUConfig
    ) where T<:Float64

Apply phase advance to particles on GPU with energy-dependent slip factor.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `γ0::T`: Relativistic gamma
- `mass::T`: Particle mass
- `α_c::T`: Momentum compaction factor
- `harmonic::Int`: Harmonic number
- `β0::T`: Relativistic beta
- `E0::T`: Reference energy
- `rf_factor::T`: RF factor
- `ϕs::T`: Synchronous phase
- `gpu_config::GPUConfig`: GPU kernel configuration
"""
function apply_phase_advance_dynamic_gpu!(
    gpu_data::GPUParticleData{T},
    γ0::T, mass::T, α_c::T, 
    harmonic::Int, β0::T, E0::T, rf_factor::T, ϕs::T,
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
        @cuda prefer_l1=true blocks=blocks threads=threads phase_advance_dynamic_kernel!(
            gpu_data.z, gpu_data.ΔE, γ0, mass, α_c, harmonic, β0, E0, rf_factor, ϕs
        )
    else
        @cuda blocks=blocks threads=threads phase_advance_dynamic_kernel!(
            gpu_data.z, gpu_data.ΔE, γ0, mass, α_c, harmonic, β0, E0, rf_factor, ϕs
        )
    end
    
    return nothing
end