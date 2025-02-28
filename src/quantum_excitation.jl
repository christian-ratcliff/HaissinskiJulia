"""
quantum_excitation.jl - Quantum excitation effects

This file implements the quantum excitation effects that cause stochastic
energy fluctuations in the beam due to the quantum nature of synchrotron radiation.
"""

using Random
using LoopVectorization

"""
    quantum_excitation!(
        E0::T, 
        radius::T, 
        σ_E0::T, 
        buffers::SimulationBuffers{T},
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply quantum excitation to all particles.
"""
function quantum_excitation!(
    E0::T, 
    radius::T, 
    σ_E0::T, 
    buffers::SimulationBuffers{T},
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Calculate excitation parameter
    excitation::T = calculate_excitation_strength(E0, radius, σ_E0)
    
    # Generate random values
    randn!(buffers.random_buffer)
    
    # Apply random kicks to each particle
    n_particles = length(particles)
    @inbounds for i in 1:n_particles
        particles.coordinates.ΔE[i] += excitation * buffers.random_buffer[i]
    end
    
    return nothing
end

"""
    calculate_quantum_diffusion(E0::T, radius::T) where T<:Float64 -> T

Calculate quantum diffusion coefficient.
"""
function calculate_quantum_diffusion(E0::T, radius::T) where T<:Float64
    # Radiation constant
    Cγ::T = 8.85e-5  # [m/GeV^3]
    
    # Average energy loss per turn
    U0::T = Cγ * (E0/1e9)^4 / radius
    
    
    ∂U_∂E::T = 4 * Cγ * (E0/1e9)^3 / radius
    
    # Diffusion coefficient
    D::T = (1-(1-∂U_∂E)^2) * U0^2 / (2 * E0^2)
    
    return D
end

"""
    calculate_excitation_strength(E0::T, radius::T, σ_E0::T) where T<:Float64 -> T

Calculate quantum excitation strength parameter.
"""
function calculate_excitation_strength(E0::T, radius::T, σ_E0::T) where T<:Float64
    ∂U_∂E::T = 4 * 8.85e-5 * (E0/1e9)^3 / radius
    excitation::T = sqrt(1-(1-∂U_∂E)^2) * σ_E0
    return excitation
end