"""
quantum_excitation.jl - Quantum excitation effects

This file implements the quantum excitation effects that cause stochastic
energy fluctuations in the beam due to the quantum nature of synchrotron radiation.
"""

using Random
using LoopVectorization
using StochasticAD

"""
    quantum_excitation!(
        E0, 
        radius, 
        σ_E0::T, 
        buffers::SimulationBuffers{T},
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply quantum excitation to all particles.
Memory-efficient implementation for StochasticTriple.
"""
function quantum_excitation!(
    E0, 
    radius, 
    σ_E0::T, 
    buffers::SimulationBuffers{T},
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Check type once, outside the loop
    is_stochastic = (typeof(E0) <: StochasticTriple) || (typeof(radius) <: StochasticTriple)
    
    # Generate random values (reusing existing buffer)
    randn!(buffers.random_buffer)
    
    if is_stochastic
        # Calculate excitation parameter - single propagate call
        excitation_fn = (e0, r) -> begin
            ∂U_∂E = 4 * 8.85e-5 * (e0/1e9)^3 / r
            return sqrt(1-(1-∂U_∂E)^2) * σ_E0
        end
        
        excitation = StochasticAD.propagate(excitation_fn, E0, radius)
        
        # Apply to all particles (vectorized)
        particles.coordinates.ΔE .+= excitation .* buffers.random_buffer
    else
        # Standard implementation (vectorized)
        ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
        excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E0
        
        # Apply kicks (vectorized)
        particles.coordinates.ΔE .+= excitation .* buffers.random_buffer
    end
    
    return nothing
end