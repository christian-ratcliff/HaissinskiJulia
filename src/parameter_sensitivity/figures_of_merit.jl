"""
figures_of_merit.jl - Figure of merit implementations

This file implements the calculations for different beam quality metrics
used in sensitivity analysis.
"""

using Statistics
using StatsBase

"""
    compute_fom(fom::EnergySpreadFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64

Compute energy spread figure of merit.
"""
function compute_fom(fom::EnergySpreadFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64
    σ_E, σ_z, E0 = results
    return σ_E
end

"""
    compute_fom(fom::BunchLengthFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64

Compute bunch length figure of merit.
"""
function compute_fom(fom::BunchLengthFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64
    σ_E, σ_z, E0 = results
    return σ_z
end

"""
    compute_fom(fom::EmittanceFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64

Compute emittance figure of merit.
"""
function compute_fom(fom::EmittanceFoM, particles::StructArray{Particle{T}}, results::Tuple) where T<:Float64
    # Get the current spreads
    σ_E, σ_z, E0 = results
    
    # Calculate the correlation

    correlation = cor(particles.coordinates.z, particles.coordinates.ΔE)
    
    # Calculate the emittance
    emittance = σ_z * σ_E * sqrt(1 - correlation^2)
    
    return emittance
end