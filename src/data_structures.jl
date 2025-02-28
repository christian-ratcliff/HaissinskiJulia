"""
data_structures.jl - Core data structures for beam simulations

This file defines the fundamental data structures used in the simulation:
- Coordinate: Longitudinal phase space coordinates
- Particle: Particle representation
- BeamTurn: Container for particle states across turns
- SimulationParameters: Complete simulation parameters
- SimulationBuffers: Pre-allocated buffers for efficient computation
"""

using StructArrays
using StaticArrays

"""
    Coordinate{T} <: FieldVector{2, T}

Coordinate in longitudinal phase space.

# Fields
- `z::T`: Longitudinal position relative to reference particle
- `ΔE::T`: Energy deviation from reference energy
"""
struct Coordinate{T} <: FieldVector{2, T}
    z::T
    ΔE::T
end

"""
    Particle{T} <: FieldVector{1, Coordinate}

Simple particle representation with coordinates in phase space.

# Fields
- `coordinates::Coordinate{T}`: Current position in phase space
"""
struct Particle{T} <: FieldVector{2, Coordinate}
    coordinates::Coordinate{T}
    uncertainty::Coordinate{T}
end

"""
    BeamTurn{T}

Container for particle states across multiple turns.

# Fields
- `states::Vector{StructArray{Particle{T}}}`: Array of particle states for each turn
"""
struct BeamTurn{T<:Float64}
    states::Vector{StructArray{Particle{T}}}
end

"""
    SimulationParameters{T<:Float64}

Container for all simulation parameters.

# Fields
- `E0::T`: Reference energy [eV]
- `mass::T`: Particle mass [eV/c²]
- `voltage::T`: RF voltage [V]
- `harmonic::Int`: RF harmonic number
- `radius::T`: Accelerator radius [m]
- `pipe_radius::T`: Beam pipe radius [m]
- `α_c::T`: Momentum compaction factor
- `ϕs::T`: Synchronous phase [rad]
- `freq_rf::T`: RF frequency [Hz]
- `n_turns::Int`: Number of turns to simulate
- `use_wakefield::Bool`: Enable wakefield effects
- `update_η::Bool`: Update slip factor
- `update_E0::Bool`: Update reference energy
- `SR_damping::Bool`: Enable synchrotron radiation damping
- `use_excitation::Bool`: Enable quantum excitation
"""
mutable struct SimulationParameters{T} <: FieldVector{15, Coordinate}
    E0::T
    mass::T
    voltage::T
    harmonic::Int
    radius::T
    pipe_radius::T
    α_c::T
    ϕs::T
    freq_rf::T
    n_turns::Int
    use_wakefield::Bool
    update_η::Bool
    update_E0::Bool
    SR_damping::Bool
    use_excitation::Bool
end

"""
    SimulationBuffers{T<:Float64}

Pre-allocated buffers for efficient computation during simulation.

# Fields
- `WF::Vector{T}`: Buffer for wakefield calculations
- `potential::Vector{T}`: Buffer for potential energy calculations
- `Δγ::Vector{T}`: Buffer for gamma factor deviations
- `η::Vector{T}`: Buffer for slip factor calculations
- `coeff::Vector{T}`: Buffer for temporary coefficients
- `temp_z::Vector{T}`: General temporary storage for z coordinates
- `temp_ΔE::Vector{T}`: General temporary storage for energy deviations
- `temp_ϕ::Vector{T}`: General temporary storage for phases
- `WF_temp::Vector{T}`: Temporary wakefield values
- `λ::Vector{T}`: Line charge density values
- `convol::Vector{Complex{T}}`: Convolution results
- `ϕ::Vector{T}`: Phase values
- `random_buffer::Vector{T}`: Buffer for random numbers
"""
struct SimulationBuffers{T<:Float64}
    WF::Vector{T}
    potential::Vector{T}
    Δγ::Vector{T}
    η::Vector{T}
    coeff::Vector{T}
    temp_z::Vector{T}
    temp_ΔE::Vector{T}
    temp_ϕ::Vector{T}
    WF_temp::Vector{T}
    λ::Vector{T}
    convol::Vector{Complex{T}}
    ϕ::Vector{T}
    random_buffer::Vector{T}
end
