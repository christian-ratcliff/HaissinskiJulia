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
struct Particle{T} <: FieldVector{1, Coordinate}
    coordinates::Coordinate{T}
    # uncertainty::Coordinate{T}
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
    SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}

Type-stable container for all simulation parameters with multiple type parameters
to allow individual parameters to be StochasticTriple when needed.

# Type Parameters
- `TE`: Type of E0 (energy)
- `TM`: Type of mass
- `TV`: Type of voltage
- `TR`: Type of radius
- `TPR`: Type of pipe_radius
- `TA`: Type of α_c (momentum compaction)
- `TPS`: Type of ϕs (synchronous phase)
- `TF`: Type of freq_rf (RF frequency)

# Fields
- `E0::TE`: Reference energy [eV]
- `mass::TM`: Particle mass [eV/c²]
- `voltage::TV`: RF voltage [V]
- `harmonic::Int`: RF harmonic number
- `radius::TR`: Accelerator radius [m]
- `pipe_radius::TPR`: Beam pipe radius [m]
- `α_c::TA`: Momentum compaction factor
- `ϕs::TPS`: Synchronous phase [rad]
- `freq_rf::TF`: RF frequency [Hz]
- `n_turns::Int`: Number of turns to simulate
- `use_wakefield::Bool`: Enable wakefield effects
- `update_η::Bool`: Update slip factor
- `update_E0::Bool`: Update reference energy
- `SR_damping::Bool`: Enable synchrotron radiation damping
- `use_excitation::Bool`: Enable quantum excitation
"""
struct SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}
    E0::TE
    mass::TM
    voltage::TV
    harmonic::Int
    radius::TR
    pipe_radius::TPR
    α_c::TA
    ϕs::TPS
    freq_rf::TF
    n_turns::Int
    use_wakefield::Bool
    update_η::Bool
    update_E0::Bool
    SR_damping::Bool
    use_excitation::Bool
end

# Convenience constructor for all Float64 parameters
function SimulationParameters(E0::Float64, mass::Float64, voltage::Float64, 
                             harmonic::Int, radius::Float64, pipe_radius::Float64, 
                             α_c::Float64, ϕs::Float64, freq_rf::Float64, 
                             n_turns::Int, use_wakefield::Bool, update_η::Bool, 
                             update_E0::Bool, SR_damping::Bool, use_excitation::Bool)
    return SimulationParameters{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}(
        E0, mass, voltage, harmonic, radius, pipe_radius, 
        α_c, ϕs, freq_rf, n_turns, use_wakefield, 
        update_η, update_E0, SR_damping, use_excitation
    )
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
    # New buffers for optimized calculations
    normalized_λ::Vector{T}      # For wakefield calculations
    fft_buffer1::Vector{Complex{T}} # For in-place FFT operations
    fft_buffer2::Vector{Complex{T}} # For in-place FFT operations
    real_buffer::Vector{T}       # For storing real parts
    bin_counts::Vector{Int}      # For histogram calculations
    thread_local_buffers::Vector{Dict{Symbol, Any}}
end
