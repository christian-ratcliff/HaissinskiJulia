"""
    Module: Beam Evolution Simulation

This module implements a high-performance beam evolution simulation for particle accelerators,
including functionality for particle generation, evolution tracking, and data visualization.

# Main Components:
- Core Data Structures: ParticleState, BeamTurn, SimulationBuffers
- High-Level Simulation Functions: longitudinal_evolve, generate_particles
- Data Management Functions: write/read particle evolution data
- Physics Calculation Functions: make_separatrix, apply_wakefield
- Visualization Functions: Various plotting and animation utilities

# Physical Constants:
- SPEED_LIGHT: Speed of light in vacuum (m/s)
- ELECTRON_CHARGE: Elementary charge (C)
- MASS_ELECTRON: Electron rest mass (eV/c²)
- INV_SQRT_2π: 1/√(2π) (optimization constant)
"""

using Distributions ;
using Random ;
Random.seed!(12345) ;
using LaTeXStrings ;
using Interpolations ;
using Roots ;
using BenchmarkTools ;
using Base.Threads ;
using StaticArrays ;
using SIMD ;
using StructArrays ;
using CircularArrays ;
using PreallocationTools ;
using LinearAlgebra ;
using LoopVectorization ;
using CairoMakie ;
using KernelDensity ;
using Statistics ;
using StatsBase ;
using FFTW ;
using Colors ;
using VideoIO ;
using Dierckx ;
using Dates ;
using HDF5 ;
using Distributed ;
using OffsetArrays ;
using ColorSchemes
using ThreadsX ;
using FLoops ; 
using FHist ; 
using ProgressMeter ; 
using ProfileSVG ;
using Profile ;
using DataStructures ;
using FileIO ; 

# Physical constants
const SPEED_LIGHT = 299792458
const ELECTRON_CHARGE = 1.602176634e-19
const MASS_ELECTRON = 0.51099895069e6
const INV_SQRT_2π = 1 / sqrt(2 * π)
const ħ = 6.582119569e-16

#=
Core Data Structures
=#

"""
    ParticleState{T<:AbstractFloat}

Immutable structure representing the state of a single particle in the beam.

# Fields
- `z::T`: Longitudinal position relative to the reference particle
- `ΔE::T`: Energy deviation from the reference energy
- `ϕ::T`: Phase relative to the RF wave

# Example
```julia
particle = ParticleState(0.1, 1e-3, 0.5)  # position, energy deviation, phase
```
"""
struct ParticleState{T<:AbstractFloat}
    z::T
    ΔE::T
    ϕ::T
end

"""
    BeamTurn{T,N}

Container for particle states across multiple turns, optimized for efficient memory access.

# Fields
- `states::Array{StructArray{ParticleState{T}}, 1}`: Array of particle states for each turn

# Type Parameters
- `T`: Floating-point precision type
- `N`: Number of turns plus one (includes initial state)

# Example
```julia
beam = BeamTurn{Float64}(1000, 10000)  # 1000 turns, 10000 particles
```
"""
struct BeamTurn{T,N}
    states::Array{StructArray{ParticleState{T}}, 1}
end

"""
    SimulationBuffers{T<:AbstractFloat}

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

Used internally to minimize memory allocations during simulation steps.
"""
struct SimulationBuffers{T<:AbstractFloat}
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
end


"""
    BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T -> BeamTurn{T,N}

Constructor for BeamTurn object to store particle states across multiple turns.

# Arguments
- `n_turns::Integer`: Number of turns to simulate
- `n_particles::Integer`: Number of particles

# Returns
- `BeamTurn{T,N}`: BeamTurn object with pre-allocated storage

# Example
```julia
beam = BeamTurn{Float64}(1000, 10000)  # For 1000 turns, 10000 particles
```
"""
function BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T
    states = SVector{n_turns+1}(
        StructArray{ParticleState{T}}((
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles)
        )) for _ in 1:n_turns+1
    )
    return BeamTurn{T, n_turns+1}(states)
end


#=
High-Level Simulation Functions
=#

"""
    generate_particles(μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
                      energy::T, mass::T, ϕs::T, freq_rf::T) where T<:AbstractFloat 
                      -> StructArray{ParticleState{T}}

Generate initial particle distribution using multivariate normal sampling.

# Arguments
- `μ_z::T`: Mean longitudinal position
- `μ_E::T`: Mean energy deviation
- `σ_z::T`: Position spread (standard deviation)
- `σ_E::T`: Energy spread (standard deviation)
- `num_particles::Int`: Number of particles to generate
- `energy::T`: Reference beam energy
- `mass::T`: Particle mass
- `ϕs::T`: Synchronous phase
- `freq_rf::T`: RF frequency

# Returns
- `StructArray{ParticleState{T}}`: Initial particle states

# Implementation Notes
- Uses two-stage sampling for accurate correlations:
  1. Initial small sample to determine correlations
  2. Full distribution generation using computed covariance
- Computes RF phase from position using relativistic factors
- Thread-safe random number generation
- Efficient memory allocation using StructArrays

# Example
```julia
# Generate 10000 particles
particles = generate_particles(0.0, 0.0, 1e-3, 1e-4, 10000, 
                             3e9, mass_electron, 0.0, 500e6)
```
"""
function generate_particles(
    μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
    energy::T, mass::T, ϕs::T, freq_rf::T
) where T<:AbstractFloat
    
    # Initial sampling for correlation estimation
    initial_sample_size = min(10000, num_particles)
    z_samples = Vector{T}(undef, initial_sample_size)
    E_samples = Vector{T}(undef, initial_sample_size)
    
    # Create initial distributions
    z_dist = Normal(μ_z, σ_z)
    E_dist = Normal(μ_E, σ_E)
    
    # Generate initial samples efficiently
    rand!(z_dist, z_samples)
    rand!(E_dist, E_samples)
    
    # Calculate covariance matrix efficiently
    Σ = @views begin
        cov_zz = cov(z_samples, z_samples)
        cov_zE = cov(z_samples, E_samples)
        cov_EE = cov(E_samples, E_samples)
        [cov_zz cov_zE; cov_zE cov_EE]
    end
    
    # Create multivariate distribution
    μ = SVector{2,T}(μ_z, μ_E)
    dist_total = MvNormal(μ, Symmetric(Σ))
    
    # Pre-allocate particle states array
    particle_states = StructArray{ParticleState{T}}((
        Vector{T}(undef, num_particles),  # z
        Vector{T}(undef, num_particles),  # ΔE
        Vector{T}(undef, num_particles)   # ϕ
    ))
    
    # Calculate relativistic factors
    γ = energy / mass
    β = sqrt(1 - 1/γ^2)
    rf_factor = freq_rf * 2π / (β * SPEED_LIGHT)

    # Generate particles using thread-safe RNG
    local_rng = Random.default_rng()
    
    for i in eachindex(particle_states.z)
        sample_vec = rand(local_rng, dist_total)
        particle_states.z[i] = sample_vec[1]
        particle_states.ΔE[i] = sample_vec[2]
        particle_states.ϕ[i] = -(particle_states.z[i] * rf_factor - ϕs)
    end
    
    return particle_states, σ_E, σ_z, energy
end

"""
    apply_wakefield_inplace!(particle_states::StructArray{ParticleState{T}}, 
                           buffers::SimulationBuffers{T}, wake_factor::T, 
                           wake_sqrt::T, cτ::T, E0::T, acc_radius::T, 
                           n_particles::Int, current::T, σ_z::T) where T<:AbstractFloat -> Nothing

Apply wakefield effects to particle distribution using FFT-based convolution.

# Arguments
- `particle_states`: Current particle states
- `buffers`: Pre-allocated calculation buffers
- `wake_factor`: Wakefield strength factor
- `wake_sqrt`: Square root of wakefield wavenumber
- `cτ`: Characteristic time
- `E0`: Reference beam energy
- `acc_radius`: Accelerator radius
- `n_particles`: Number of particles
- `current`: Beam current
- `σ_z`: Bunch length

# Implementation Notes
## Algorithm Steps:
1. Calculate wake function for each particle
2. Compute line charge density using Gaussian smoothing
3. Perform FFT-based convolution of wake and density
4. Interpolate results back to particle positions
5. Update particle energies based on wakefield potential

## Optimizations:
- Uses pre-allocated buffers to minimize memory allocation
- Implements SIMD operations via @turbo macro
- Uses power-of-2 FFT sizes for efficiency
- Applies linear interpolation for final mapping
- Includes bounds checking optimization

## Physical Model:
- Implements resonator wakefield model
- Includes proper scaling with beam parameters
- Handles both short and long-range effects

# Example
```julia
apply_wakefield_inplace!(states, buffers, 1e6, sqrt(2e1), 4e-3,
                        3e9, 100.0, 10000, 1e-3, 1e-3)
```
"""
function apply_wakefield_inplace!(
    particle_states::StructArray{ParticleState{T}}, 
    buffers::SimulationBuffers{T}, 
    wake_factor::T, 
    wake_sqrt::T, 
    cτ::T,  
    n_particles::Int,
    current::T,
    σ_z::T,
    bin_edges::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}
    ) where T<:AbstractFloat
    
    
    if !all(iszero, buffers.λ)
        fill!(buffers.λ, zero(T))
    end
    if !all(iszero, buffers.WF_temp)
        fill!(buffers.WF_temp, zero(T))
    end
    if !all(iszero, buffers.convol)
        fill!(buffers.convol, zero(T))
    end
    # if !all(iszero, buffers.WF)
    #     fill!(buffers.WF, zero(T))
    # end
    # if !all(iszero, buffers.potential)
    #     fill!(buffers.potential, zero(T))
    # end

    # buffers.WF .= zero(T)
    # buffers.potential .= zero(T)
    # buffers.WF_temp .= zero(T)
    # buffers.λ .= zero(T)
    # buffers.convol .= zero(T)

    z_positions = @views particle_states.z
    inv_cτ = 1 / cτ
    
    # Calculate optimal bin count for histogram
    # nbins = next_power_of_two(Int(10^(ceil(Int, log10(length(z_positions))-2))))

    # Calculate histogram
    bin_centers, bin_amounts = calculate_histogram(z_positions, bin_edges)
    nbins = length(bin_centers)
    power_2_length = nbins  * 2 #next_power_of_two(2*nbins-1)
    
    
    # Calculate wake function for each bin
    @turbo for i in eachindex(bin_centers)
        z = bin_centers[i]
        buffers.WF_temp[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    # Calculate line charge density using Gaussian smoothing
    delta_std = (15 * σ_z) / σ_z / 100
    @turbo for i in eachindex(bin_centers)
        buffers.λ[i] = delta(bin_centers[i], delta_std)
    end
    
    # Prepare arrays for convolution
    normalized_amounts = bin_amounts .* (1/n_particles)
    λ = buffers.λ[1:nbins]
    WF_temp = buffers.WF_temp[1:nbins]
    convol = buffers.convol[1:power_2_length]
    
    # Perform convolution and scale by current
    convol .= FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length) .* current
    
    # Interpolate results back to particle positions
    temp_z = range(minimum(z_positions), maximum(z_positions), length=length(convol))
    resize!(buffers.potential, length(z_positions))
    buffers.potential .= LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line()).(z_positions)
    
    # Update particle energies and calculate wake function
    @turbo for i in eachindex(z_positions)
        z = z_positions[i]
        particle_states.ΔE[i] -= buffers.potential[i]
        buffers.WF[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    return nothing
end


"""
    longitudinal_evolve(n_turns::Int, particle_states::StructArray{ParticleState{T}},
                       ϕs::T, α_c::T, mass::T, voltage::T, harmonic::Int,
                       acc_radius::T, freq_rf::T, pipe_radius::T, E0::T, σ_E::T;
                       kwargs...) where T<:AbstractFloat -> 
                       Union{BeamTurn{T,N}, Tuple{BeamTurn{T,N}, Vector{Any}}, 
                       Tuple{BeamTurn{T,N}, Vector{Any}, Vector{Any}}}

Simulate longitudinal beam evolution with complete trajectory storage, collective effects, and visualization options.

# Arguments
## Required Parameters
- `n_turns::Int`: Number of turns to simulate
- `particle_states::StructArray{ParticleState{T}}`: Initial particle distribution
- `ϕs::T`: Synchronous phase [rad]
- `α_c::T`: Momentum compaction factor
- `mass::T`: Particle mass [eV/c²]
- `voltage::T`: RF voltage [V]
- `harmonic::Int`: RF harmonic number
- `acc_radius::T`: Accelerator radius [m]
- `freq_rf::T`: RF frequency [Hz]
- `pipe_radius::T`: Beam pipe radius [m]
- `E0::T`: Reference energy [eV]
- `σ_E::T`: Energy spread [eV]

## Keyword Arguments
- `update_η::Bool=false`: Update slip factor each turn based on particle energy
- `update_E0::Bool=false`: Update reference energy each turn due to acceleration
- `SR_damping::Bool=false`: Include synchrotron radiation damping
- `use_excitation::Bool=false`: Include quantum excitation effects
- `use_wakefield::Bool=false`: Include collective wakefield effects
- `plot_potential::Bool=false`: Generate plots of wakefield potential
- `plot_WF::Bool=false`: Generate plots of wake function
- `write_to_file::Bool=false`: Save particle data to HDF5 file
- `output_file::String="particles_output.hdf5"`: Output file path
- `additional_metadata::Dict{String,Any}=Dict{String,Any}()`: Additional simulation parameters to save

# Returns
- Without plots: `BeamTurn{T,N}` containing complete particle trajectory
- With potential plots: Tuple of `BeamTurn{T,N}` and vector of potential plots
- With both plots: Tuple of `BeamTurn{T,N}`, potential plots, and wake function plots

# Physics Implementation
## Single-Particle Dynamics
1. RF Cavity:
   - Energy kick: ΔE += eV(sin(ϕ) - sin(ϕs))
   - Phase advance: Δϕ = 2πh η/(β²E₀) ΔE
   - Position update: z = -(ϕ - ϕs)/(2πf_rf/(βc))

2. Synchrotron Radiation:
   - Energy loss per turn: U = 8.85e-5 * (E/1e9)⁴ / R
   - Damping: ΔE *= (1 - ∂U/∂E)
   - Quantum excitation: ΔE += √(1-(∂U/∂E)²)σ_E * randn()

## Collective Effects
1. Wakefield:
   - Resonator wake model
   - FFT-based convolution
   - Energy kick from wake potential

2. Reference Energy:
   - Optional acceleration
   - Dynamic slip factor
   - Phase space updates

# Implementation Notes
## Memory Management
1. Trajectory Storage:
   - Complete turn-by-turn storage
   - Efficient StructArray usage
   - Pre-allocated buffers

2. File I/O:
   - Chunked HDF5 storage
   - Metadata preservation
   - Timestamp organization

3. Optimization Features:
   - SIMD operations
   - Thread safety
   - Minimal allocations
   - Efficient array access

# Example Usage
```julia
# Basic evolution
trajectory = longitudinal_evolve(
    1000,                # number of turns
    initial_states,      # initial distribution
    0.0,                # synchronous phase
    1.89e-4,            # momentum compaction
    0.511e6,            # electron mass
    1.0e6,              # RF voltage
    1320,               # harmonic number
    100.0,              # accelerator radius
    500e6,              # RF frequency
    0.02,               # pipe radius
    3.0e9,              # reference energy
    1.0e6               # energy spread
)

# With effects and visualization
trajectory, pot_plots, wf_plots = longitudinal_evolve(
    1000, initial_states, 0.0, 1.89e-4, 0.511e6, 1.0e6, 1320,
    100.0, 500e6, 0.02, 3.0e9, 1.0e6;
    update_η=true,
    SR_damping=true,
    use_wakefield=true,
    plot_potential=true,
    plot_WF=true
)
```

See also: [`longitudinal_evolve!`](@ref) for in-place evolution.
"""

function longitudinal_evolve(
    n_turns::Int,
    particle_states::StructArray{ParticleState{T}},
    ϕs::T,
    α_c::T,
    mass::T,
    voltage::T,
    harmonic::Int,
    acc_radius::T,
    freq_rf::T,
    pipe_radius::T,
    E0::T,
    σ_E::T,
    σ_z::T;
    update_η::Bool=false,
    update_E0::Bool=false,
    SR_damping::Bool=false,
    use_excitation::Bool=false,
    use_wakefield::Bool=false, 
    plot_potential::Bool=false,
    plot_WF::Bool=false,
    write_to_file::Bool=false,
    output_file::String="particles_output.hdf5",
    additional_metadata::Dict{String, Any}=Dict{String, Any}()
) where T<:AbstractFloat

    # Pre-compute constants
    γ0 = E0 / mass
    β0 = sqrt(1 - 1/γ0^2)
    η0 = α_c - 1/(γ0^2)
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Initialize sizes and buffers
    n_particles = length(particle_states.z)
    buffers = create_simulation_buffers(n_particles, Int(n_particles/10), T)
    
    particle_trajectory = BeamTurn{T}(n_turns, n_particles)
    @views particle_trajectory.states[1] .= particle_states

    nbins = next_power_of_two(Int(10^(ceil(Int, log10(length(particle_states.z))-2))))
    bin_edges = range(-7.5*σ_z, 7.5*σ_z, length=nbins+1)

    σ_E_buffer = CircularBuffer{T}(50)
    E0_buffer = CircularBuffer{T}(50)
    z_buffer = CircularBuffer{T}(50)
    push!(σ_E_buffer,σ_E)
    push!(E0_buffer,E0)
    push!(z_buffer,σ_z)

    # Initialize file and write metadata if writing is enabled
    if write_to_file
        timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
        folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(n_particles)"
        folder_storage = joinpath(folder_storage, timestamp)
        mkpath(folder_storage)
        
        output_file = joinpath(folder_storage, output_file)

        h5open(output_file, "w") do file
            # Create datasets with chunking for efficient writing
            chunk_size = min(n_particles, 10000)
            z_dset = create_dataset(file, "z", T, ((n_particles, n_turns + 1)), 
                                chunk=(chunk_size, 1))
            phi_dset = create_dataset(file, "phi", T, ((n_particles, n_turns + 1)), 
                                    chunk=(chunk_size, 1))
            dE_dset = create_dataset(file, "dE", T, ((n_particles, n_turns + 1)), 
                                chunk=(chunk_size, 1))
            
            # Write initial state
            z_dset[:, 1] = particle_states.z
            phi_dset[:, 1] = particle_states.ϕ
            dE_dset[:, 1] = particle_states.ΔE
            
            # Create metadata group
            meta_group = create_group(file, "metadata")
            
            # Write simulation parameters
            simulation_metadata = Dict{String, Any}(
                "n_turns" => n_turns,
                "n_particles" => n_particles,
                "sync_phase" => ϕs,
                "alpha_c" => α_c,
                "mass" => mass,
                "voltage" => voltage,
                "harmonic" => harmonic,
                "acc_radius" => acc_radius,
                "freq_rf" => freq_rf,
                "pipe_radius" => pipe_radius,
                "E0" => E0,
                "sigma_E" => σ_E,
                "update_eta" => update_η,
                "update_E0" => update_E0,
                "SR_damping" => SR_damping,
                "use_excitation" => use_excitation,
                "use_wakefield" => use_wakefield,
                "gamma0" => γ0,
                "beta0" => β0,
                "eta0" => η0,
                "timestamp" => string(Dates.now())
            )
            
            # Merge with additional metadata if provided
            merge!(simulation_metadata, additional_metadata)
            
            # Write all metadata
            for (key, value) in simulation_metadata
                meta_group[key] = value
            end
        end
    end

    # Initialize wakefield parameters if needed
    if use_wakefield
        kp = T(3e1)
        Z0 = T(120π)
        cτ = T(4e-3)
        wake_factor = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt = sqrt(2*kp/pipe_radius)
    end

    # Initialize plot vectors if needed
    potential_plots = plot_potential ? Vector{Any}(undef, n_turns) : nothing
    WF_plots = plot_WF ? Vector{Any}(undef, n_turns) : nothing
    
    # Setup progress meter
    p = Progress(n_turns, desc="Simulating Turns: ")

    # Main evolution loop
    @inbounds for turn in 1:n_turns
        σ_E = std(particle_states.ΔE)
        σ_z = std(particle_states.z)
        # RF voltage kick
        @turbo for i in 1:n_particles
            particle_states.ΔE[i] += voltage * (sin(particle_states.ϕ[i]) - sin_ϕs)
        end
        
        # Quantum excitation
        if use_excitation
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E
            randn!(buffers.potential)
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] += excitation * buffers.potential[i]
                
            end
        end
        
        
        # Synchrotron radiation damping
        if SR_damping
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] *= (1 - ∂U_∂E)
            end
        end
        
        # Update reference energy if needed
        if update_E0
            E0 += voltage * sin_ϕs
            γ0 = E0/mass 
            β0 = sqrt(1 - 1/γ0^2)
            if update_E0
                E0 -= ∂U_∂E * E0 / 4
                γ0 = E0/mass 
                β0 = sqrt(1 - 1/γ0^2)
            end
        end
        
        # Update phase advance
        if update_η
            @turbo for i in 1:n_particles
                buffers.Δγ[i] = particle_states.ΔE[i] / mass
                buffers.η[i] = α_c - 1/(γ0 + buffers.Δγ[i])^2
                buffers.coeff[i] = 2π * harmonic * buffers.η[i] / (β0 * β0 * E0)
                particle_states.ϕ[i] += buffers.coeff[i] * particle_states.ΔE[i]
            end
        else
            coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
            @turbo for i in 1:n_particles
                particle_states.ϕ[i] += coeff * particle_states.ΔE[i]
            end
        end
        
        # Update longitudinal positions
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        @turbo for i in 1:n_particles
            particle_states.z[i] = (-particle_states.ϕ[i] + ϕs) / rf_factor
        end
    
        # Apply wakefield effects if enabled
        if use_wakefield
            ν_s = sqrt(voltage * harmonic * α_c / (2π * E0))
            curr = (abs(η0) / η0) * ELECTRON_CHARGE / (2 * π * ν_s * σ_E) * 
                   (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
            
            if plot_potential && plot_WF
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                # Store potential plot
                fig = Figure(size=(800, 500))
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                Label(fig[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                scatter!(ax,
                    particle_states.z / σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig

                # Store wakefield plot
                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                Label(fig2[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2
                
            elseif plot_WF && !plot_potential
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                Label(fig2[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                xlims!(ax2, -.02, 0)
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2

            elseif plot_potential && !plot_WF
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                fig = Figure(size=(800, 500))
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                Label(fig[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                scatter!(ax,
                    particle_states.z /σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig
            else
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
            end
        end
        
        
        
        

        # Store current state in trajectory
        assign_to_turn!(particle_trajectory, particle_states, turn+1)

        # Write to file if enabled
        if write_to_file
            h5open(output_file, "r+") do file
                file["z"][:, turn + 1] = particle_states.z
                file["phi"][:, turn + 1] = particle_states.ϕ
                file["dE"][:, turn + 1] = particle_states.ΔE
            end
        end
        push!(σ_E_buffer,σ_E)
        push!(E0_buffer,E0)
        push!(z_buffer,σ_z)
        if abs(mean(σ_E_buffer)/mean(E0_buffer) - std(particle_states.ΔE)/E0) < 1e-9
            println("Converged at turn $turn with σ_E = ", mean(σ_E_buffer))
            σ_E = mean(σ_E_buffer)
            σ_z = mean(z_buffer)
            E0 = mean(E0_buffer)
            resize!(particle_trajectory.states, turn+1)
            return particle_trajectory, σ_E, σ_z, E0
        end
        
        next!(p)
    end
    
    # Return appropriate output based on plotting options
    if plot_potential && plot_WF
        return particle_trajectory, potential_plots, WF_plots, σ_E, σ_z
    elseif plot_potential && !plot_WF
        return particle_trajectory, potential_plots, σ_E, σ_z
    elseif !plot_potential && plot_WF
        return particle_trajectory, WF_plots, σ_E, σ_z
    else
        return particle_trajectory, σ_E, σ_z, E0
    end
end

"""
    longitudinal_evolve!(n_turns::Int, particle_states::StructArray{ParticleState{T}},
                        ϕs::T, α_c::T, mass::T, voltage::T, harmonic::Int,
                        acc_radius::T, freq_rf::T, pipe_radius::T, E0::T, σ_E::T;
                        update_η::Bool=false,
                        update_E0::Bool=false,
                        SR_damping::Bool=false,
                        use_excitation::Bool=false,
                        use_wakefield::Bool=false,
                        display_counter::Bool=true,
                        plot_scatter::Bool=false
                        ) where T<:AbstractFloat -> Union{Nothing, Vector{Any}}

In-place particle beam evolution simulation with minimal memory overhead.

# Arguments
## Required Parameters
- `n_turns::Int`: Number of turns to simulate
- `particle_states::StructArray{ParticleState{T}}`: Particle distribution (modified in-place)
- `ϕs::T`: Synchronous phase [rad]
- `α_c::T`: Momentum compaction factor
- `mass::T`: Particle mass [eV/c²]
- `voltage::T`: RF voltage [V]
- `harmonic::Int`: RF harmonic number
- `acc_radius::T`: Accelerator radius [m]
- `freq_rf::T`: RF frequency [Hz]
- `pipe_radius::T`: Beam pipe radius [m]
- `E0::T`: Reference energy [eV]
- `σ_E::T`: Energy spread [eV]

## Keyword Arguments
- `update_η::Bool=false`: Enable slip factor updates
- `update_E0::Bool=false`: Enable reference energy updates
- `SR_damping::Bool=false`: Include synchrotron radiation damping
- `use_excitation::Bool=false`: Include quantum excitation
- `use_wakefield::Bool=false`: Include wakefield effects
- `display_counter::Bool=true`: Show progress counter
- `plot_scatter::Bool=false`: Generate phase space plots

# Returns
- `Nothing` if plot_scatter=false
- `Vector{Any}` of scatter plots if plot_scatter=true

# Implementation Notes
## Memory Management
- No trajectory storage
- Fixed buffer allocation
- In-place array operations
- SIMD optimizations

## Physics Features
1. RF Cavity Dynamics:
   - Energy kick: ΔE += eV(sin(ϕ) - sin(ϕs))
   - Phase advance: Δϕ = 2πh η/(β²E₀) ΔE
   - Position update: z = -(ϕ - ϕs)/(2πf_rf/(βc))

2. Radiation Effects:
   - Energy loss per turn: U = 8.85e-5 * (E/1e9)⁴ / R
   - Quantum excitation: ΔE += √(1-(∂U/∂E)²)σ_E * randn()

3. Collective Effects:
   - Wakefield convolution
   - Space charge (through wakefield)
   - Dynamic energy updating

## Visualization Features
- Phase space plots (ϕ vs ΔE/σ_E)
- Separatrix overlay
- Turn-by-turn updates
"""
function longitudinal_evolve!(
    n_turns::Int,
    particle_states::StructArray{ParticleState{T}},
    ϕs::T,
    α_c::T,
    mass::T,
    voltage::T,
    harmonic::Int,
    acc_radius::T,
    freq_rf::T,
    pipe_radius::T,
    E0::T,
    σ_E::T,
    σ_z::T;
    update_η::Bool=false,
    update_E0::Bool=false,
    SR_damping::Bool=false,
    use_excitation::Bool=false,
    use_wakefield::Bool=false,
    display_counter::Bool=true,
    plot_scatter::Bool=false,
    plot_potential::Bool=false,
    plot_WF::Bool=false
) where T<:AbstractFloat
    
    # Pre-compute physical constants
    γ0 = E0 / mass
    β0 = sqrt(1 - 1/γ0^2)
    η0 = α_c - 1/(γ0^2)
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # temp_std_vec = Any[]
# 
    # Initialize buffers
    n_particles = length(particle_states.z)
    buffers = create_simulation_buffers(n_particles, Int(n_particles/10), T)

    nbins = next_power_of_two(Int(10^(ceil(Int, log10(length(particle_states.z))-2))))
    bin_edges = range(-7.5*σ_z, 7.5*σ_z, length=nbins+1)
    
    σ_E_buffer = CircularBuffer{T}(50)
    E0_buffer = CircularBuffer{T}(50)
    z_buffer = CircularBuffer{T}(50)
    push!(σ_E_buffer,σ_E)
    push!(E0_buffer,E0)
    push!(z_buffer,σ_z)

    # Setup for scatter plots if requested
    scatter_plots = plot_scatter ? Vector{Any}(undef, n_turns+1) : nothing
    potential_plots = plot_potential ? Vector{Any}(undef, n_turns+1) : nothing
    WF_plots = plot_WF ? Vector{Any}(undef, n_turns+1) : nothing
    if plot_scatter
        boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
        boundary_obs = Observable((boundary_points[1], boundary_points[2]))
        
        # Initial plot
        fig = Figure(size=(800, 500))
        ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
        Label(fig[1, 1, Top()], "Turn 1", fontsize = 20)
        scatter!(ax, 
            particle_states.ϕ,
            particle_states.ΔE / σ_E,
            markersize = 1, color = :black)
        lines!(ax, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
        xlims!(ax, 0, 3π/2)
        ylims!(ax, minimum(boundary_points[2]) / σ_E-5, maximum(boundary_points[2]) / σ_E+5)
        scatter_plots[1] = fig
    end
    
    # Initialize wakefield parameters if needed
    if use_wakefield
        kp = T(3e1)
        Z0 = T(120π)
        cτ = T(4e-3)
        wake_factor = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt = sqrt(2*kp/pipe_radius)
    end
    
    # Setup progress meter
    if display_counter
        p = Progress(n_turns, desc="Simulating Turns: ")
    end
    old_avg = σ_E/E0

    # Main evolution loop
    @inbounds for turn in 1:n_turns
        # RF voltage kick
        σ_E = std(particle_states.ΔE)
        σ_z = std(particle_states.z)
        
        @turbo for i in 1:n_particles
            particle_states.ΔE[i] += voltage * (sin(particle_states.ϕ[i]) - sin_ϕs)
        end
        # Quantum excitation
        if use_excitation

            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E

            randn!(buffers.potential)
            for i in 1:n_particles
                particle_states.ΔE[i] += excitation * buffers.potential[i] #This is not actually the potential, merely a random number with the right distribution, I just use the buffer because its already allocated
            end
        end
        
        
        # Synchrotron radiation damping
        if SR_damping
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] *= (1 - ∂U_∂E)
            end
        end
        
        # Update reference energy if needed
        if update_E0
            E0 += voltage * sin_ϕs
            γ0 = E0/mass 
            β0 = sqrt(1 - 1/γ0^2)
            if SR_damping
                E0 -= ∂U_∂E * E0  / 4
                γ0 = E0/mass 
                β0 = sqrt(1 - 1/γ0^2)
            end
        end
        
        # Update phase advance
        if update_η
            @turbo for i in 1:n_particles
                buffers.Δγ[i] = particle_states.ΔE[i] / mass
                buffers.η[i] = α_c - 1/(γ0 + buffers.Δγ[i])^2
                buffers.coeff[i] = 2π * harmonic * buffers.η[i] / (β0 * β0 * E0)
                particle_states.ϕ[i] += buffers.coeff[i] * particle_states.ΔE[i]
            end
        else
            coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
            @turbo for i in 1:n_particles
                particle_states.ϕ[i] += coeff * particle_states.ΔE[i]
            end
        end
        
        # Update longitudinal positions
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        @turbo for i in 1:n_particles
            particle_states.z[i] = (-particle_states.ϕ[i] + ϕs) / rf_factor
        end

        # Apply wakefield effects if enabled
        if use_wakefield
            ν_s = sqrt(voltage * harmonic * α_c / (2π * E0))
            curr = (abs(η0) / η0) * ELECTRON_CHARGE / (2 * π * ν_s * σ_E) * 
                   (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
            
            if plot_potential && plot_WF
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                # Store potential plot
                fig = Figure(size=(800, 500))
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                xlims!(ax, minimum(particle_states.z / σ_z) - 0.2,  maximum(particle_states.z / σ_z)+0.2)
                # ylims!(ax, minimum(buffers.potential)-2e4,  maximum(buffers.potential)+2e4)
                Label(fig[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                scatter!(ax,
                    particle_states.z / σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig

                # Store wakefield plot
                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                Label(fig2[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2
                
            elseif plot_WF && !plot_potential
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                Label(fig2[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2

            elseif plot_potential && !plot_WF
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
                
                fig = Figure(size=(800, 500))
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                Label(fig[1, 1, Top()], "Turn $(turn)", fontsize = 20)
                xlims!(ax, -4,  4)
                # ylims!(ax, minimum(buffers.potential)-2e4,  maximum(buffers.potential)+2e4)
                scatter!(ax,
                    particle_states.z /σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig
            else
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
            end
        end
        
        
        
        # Generate scatter plot if requested
        if plot_scatter
            fig = Figure(size=(800, 500))
            ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
            Label(fig[1, 1, Top()], "Turn $(turn+1)", fontsize = 20)
            scatter!(ax, 
                particle_states.ϕ ,
                particle_states.ΔE / σ_E,
                markersize = 1, color = :black)
            scatter!(ax, 
                5*π/6,
                0,
                markersize = 10, color = :blue)
            lines!(ax, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
            xlims!(ax, 0, 3π/2)
            ylims!(ax, minimum(boundary_points[2]) / σ_E-5, maximum(boundary_points[2]) / σ_E+5)
            scatter_plots[turn+1] = fig
        end

        push!(σ_E_buffer,σ_E)
        push!(E0_buffer,E0)
        push!(z_buffer,σ_z)
        if abs(mean(σ_E_buffer)/mean(E0_buffer) - std(particle_states.ΔE)/E0) < 1e-9
            println("Converged at turn $turn with σ_E = ", mean(σ_E_buffer))
            σ_E = mean(σ_E_buffer)
            σ_z = mean(z_buffer)
            E0 = mean(E0_buffer)
            return σ_E, σ_z, E0
        end

        # Update progress bar
        if display_counter
            next!(p)
        end

    end
    ###Use multiple dispatch and turn this into many functions
    if plot_potential && plot_WF && plot_scatter
        return scatter_plots, potential_plots, WF_plots, σ_E, σ_z
    elseif plot_potential && plot_scatter && !plot_WF
        return scatter_plots, potential_plots, σ_E, σ_z
    elseif plot_WF && plot_scatter && !plot_potential
        return scatter_plots, WF_plots, σ_E, σ_z
    elseif plot_potential && plot_WF && !plot_scatter
        return potential_plots, WF_plots, σ_E, σ_z
    elseif plot_potential && !plot_WF && !plot_scatter
        return potential_plots, σ_E, σ_z
    elseif plot_WF && !plot_potential && !plot_scatter
        return WF_plots, σ_E, σ_z
    elseif plot_scatter && !plot_potential && !plot_WF
        return scatter_plots, σ_E, σ_z
    else
        return σ_E, σ_z, E0#, temp_std_vec
    end
    
end

#=
High-Level Simulation Functions
=#

"""
    threaded_fieldwise_copy!(destination, source)

Perform a threaded, field-wise copy of particle states using SIMD optimization.

# Arguments
- `destination::StructArray{ParticleState{T}}`: Destination particle states
- `source::StructArray{ParticleState{T}}`: Source particle states

# Implementation Notes
- Uses @turbo macro for SIMD optimization
- Copies z, ΔE, and ϕ fields in parallel
- Assumes equal length of source and destination arrays

# Example
```julia
threaded_fieldwise_copy!(new_states, old_states)
```
"""
function threaded_fieldwise_copy!(destination, source)
    @assert length(destination.z) == length(source.z)
    @turbo for i in 1:length(source.z)
        destination.z[i] = source.z[i]
        destination.ΔE[i] = source.ΔE[i]
        destination.ϕ[i] = source.ϕ[i]
    end
end

"""
    assign_to_turn!(particle_trajectory, particle_states, turn)

Assign particle states to a specific turn in the trajectory using threaded copy.

# Arguments
- `particle_trajectory::BeamTurn`: Complete particle trajectory
- `particle_states::StructArray{ParticleState{T}}`: Current particle states
- `turn::Integer`: Turn number to assign states to

# Example
```julia
assign_to_turn!(trajectory, current_states, 10)  # Assign states to turn 10
```
"""
function assign_to_turn!(particle_trajectory, particle_states, turn)
    threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
end

#=
Core Physics Functions
=#

"""
    delta(x::T, σ::T) where T<:AbstractFloat -> T

Calculate a Gaussian delta function for beam distribution smoothing.

# Arguments
- `x::T`: Position value
- `σ::T`: Standard deviation (smoothing parameter)

# Returns
- `T`: Smoothed delta function value at position x

# Implementation Notes
- Uses pre-computed INV_SQRT_2π for efficiency
- Implements Gaussian smoothing with standard deviation σ
- Optimized for SIMD operations

# Example
```julia
smoothed_value = delta(0.1, 0.01)
```
"""
@inline function delta(x::T, σ::T) where T<:AbstractFloat
    σ_inv = INV_SQRT_2π / σ
    exp_factor = -0.5 / (σ^2)
    return σ_inv * exp(x^2 * exp_factor)
end

"""
    FastConv1D(f::AbstractVector{T}, g::AbstractVector{T}) where T -> Vector{Complex{T}}

Compute the fast convolution of two vectors using FFT.

# Arguments
- `f::AbstractVector{T}`: First input vector
- `g::AbstractVector{T}`: Second input vector

# Returns
- `Vector{Complex{T}}`: Convolution result

# Implementation Notes
- Uses FFT for O(n log n) complexity
- Performs point-wise multiplication in frequency domain
- Returns complex values from inverse FFT

# Example
```julia
result = FastConv1D(signal1, signal2)
```
"""
@inline function FastConv1D(f::AbstractVector{T}, g::AbstractVector{T}) where T
    return ifft(fft(f).*fft(g))
end

"""
    FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int) where T

Compute linear convolution with automatic padding to power of 2 length.

# Arguments
- `f::AbstractVector{T}`: First input vector
- `g::AbstractVector{T}`: Second input vector
- `power_2_length::Int`: Desired power-of-2 length for padded vectors

# Returns
- `Vector{Complex{T}}`: Linear convolution result

# Implementation Notes
- Pads inputs to power-of-2 length for FFT efficiency
- Ensures correct convolution length
- Manages memory allocations efficiently

# Example
```julia
result = FastLinearConvolution(signal1, signal2, 1024)
```
"""
@inline function FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int) where T
    pad_and_ensure_power_of_two!(f, g, power_2_length)
    return FastConv1D(f, g)
end

"""
    is_power_of_two(n::Int) -> Bool

Check if a number is a power of two using bitwise operations.

# Arguments
- `n::Int`: Number to check

# Returns
- `Bool`: true if n is a power of 2, false otherwise

# Example
```julia
is_power_two = is_power_of_two(1024)  # returns true
```
"""
function is_power_of_two(n::Int)
    return (n & (n - 1)) == 0 && n > 0
end

"""
    next_power_of_two(n::Int) -> Int

Find the next power of two greater than or equal to n.

# Arguments
- `n::Int`: Input number

# Returns
- `Int`: Next power of two ≥ n

# Example
```julia
next_pow2 = next_power_of_two(1000)  # returns 1024
```
"""
function next_power_of_two(n::Int)
    return Int(2^(ceil(log2(n))))
end

#=
Visualization Functions
=#

"""
    precompute_densities(particles_out, σ_z, σ_E) -> Tuple{Vector{KernelDensity.UnivariateKDE}}

Pre-compute kernel density estimates for particle distributions across all turns.

# Arguments
- `particles_out::BeamTurn`: Particle trajectory data
- `σ_z::Float64`: Longitudinal beam size
- `σ_E::Float64`: Energy spread

# Returns
- Tuple of vectors containing KDE objects for z and E distributions

# Implementation Notes
- Uses parallel processing for large datasets
- Applies appropriate boundary conditions
- Optimizes number of interpolation points

# Example
```julia
z_densities, E_densities = precompute_densities(beam_data, 1e-3, 1e-4)
```
"""
function precompute_densities(particles_out, σ_z, σ_E)
    n_turns = length(particles_out)
    z_densities = Vector{KernelDensity.UnivariateKDE}(undef, n_turns)
    E_densities = Vector{KernelDensity.UnivariateKDE}(undef, n_turns)
    p = Progress(n_turns, desc="Precomputing densities: ")
    
    for i in 1:n_turns
        z_data = particles_out[i].z
        E_data = particles_out[i].ΔE
        
        z_normalized = @view(z_data[.-5 .< z_data ./ σ_z .< 5]) ./ σ_z
        E_normalized = @view(E_data[.-120 .< E_data ./ σ_E .< 120]) ./ σ_E
        
        z_densities[i] = kde(z_normalized, boundary=(-5,5), npoints=100)
        E_densities[i] = kde(E_normalized, boundary=(-120,120), npoints=200)
        next!(p)
    end
    
    return z_densities, E_densities
end

"""
    create_animation_from_pngs(plots_vector, n_particles; fps=60, filename="animation.mp4")

Create an MP4 animation from a vector of plots using parallel processing.

# Arguments
- `plots_vector::Vector`: Vector of plot objects
- `n_particles::Int`: Number of particles (for directory naming)

# Keywords
- `fps::Int=60`: Frames per second for the animation
- `filename::String="animation.mp4"`: Output filename

# Implementation Notes
- Creates temporary PNG files in parallel
- Uses multi-threading for frame generation
- Automatically manages temporary files
- Creates timestamped output directories

# Example
```julia
create_animation_from_pngs(phase_space_plots, 10000, fps=30)
```
"""
function create_animation_from_pngs(plots_vector, n_particles; fps=60, filename="animation.mp4")
    dir_frames = "frames"
    folder_storage = "Haissinski/particle_sims/turns$(length(plots_vector)-1)_particles$(n_particles)"
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    dir = joinpath(folder_storage, timestamp, dir_frames)
    mkpath(dir)
    filename = joinpath(folder_storage, timestamp, filename)
    
    p2 = Progress(length(plots_vector), desc="Generating Frames: ")
    n = length(plots_vector)
    chunks = Iterators.partition(1:n, ceil(Int, n/Threads.nthreads()))
    
    Threads.@threads for chunk in collect(chunks)
        for i in chunk
            save(joinpath(dir, "frame_$i.png"), plots_vector[i])
            next!(p2)
        end
    end
    
    fig = Figure()
    ax = Axis(fig[1,1])
    hidedecorations!(ax)
    hidespines!(ax)
    p = Progress(length(plots_vector), desc="Generating animation: ")
    
    record(fig, filename, 1:length(plots_vector), framerate=fps) do frame
        empty!(ax)
        img = load(joinpath(dir, "frame_$frame.png"))
        image!(ax, rotr90(img))
        next!(p)
    end
    
    rm(dir, recursive=true)
    println("Animation complete!")
end

"""
    all_animate_optimized(n_turns, particles_out, ϕs, α_c, mass, voltage, harmonic, E0, σ_E, σ_z, filename) -> Nothing

Generate an optimized animation of beam evolution showing phase space and distributions.

# Arguments
- `n_turns::Int64`: Number of turns to animate
- `particles_out::BeamTurn{Float64}`: Particle trajectory data
- `ϕs::Float64`: Synchronous phase
- `α_c::Float64`: Momentum compaction factor
- `mass::Float64`: Particle mass
- `voltage::Float64`: RF voltage
- `harmonic::Int64`: RF harmonic number
- `E0::Float64`: Reference energy
- `σ_E::Float64`: Energy spread
- `σ_z::Float64`: Bunch length
- `filename::String="all_animation_optimized.mp4"`: Output filename

# Implementation Notes
- Pre-computes densities for performance
- Uses static figure components
- Implements efficient memory management
- Applies optimized plotting strategies
- Creates timestamped output directories

# Example
```julia
all_animate_optimized(1000, beam_data, 0.0, 1e-3, mass_electron, 1e6, 400, 3e9, 1e-4, 1e-3)
```
"""
function all_animate_optimized(
    n_turns::Int64,
    particles_out::BeamTurn{Float64},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64,
    filename::String="all_animation_optimized.mp4")
    
    # Create output directory
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    filename = joinpath(folder_storage, filename)

    # Physics calculations
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    
    # Pre-compute densities
    # z_densities, E_densities = precompute_densities(particles_out, σ_z, σ_E)
    
    # Setup figure
    println("Setting up figure...")
    fig = Figure(;size=(1400, 900), font="Arial")
    ax_z = Axis(fig[1, 1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
    ax_E = Axis(fig[1, 2], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
    ax_phase = Axis(fig[2, 1:2], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
    title_label = Label(fig[0, :], "Turn 1", fontsize=20, halign=:center)
    
    # Set static limits
    ylims!(ax_z, (0, 0.5))
    ylims!(ax_E, (0, .5))
    xlims!(ax_z, (-5, 5))
    xlims!(ax_E, (-5, 5))
    xlims!(ax_phase, (0, 3π/2))
    
    
    boundary_obs = Observable((boundary_points[1], boundary_points[2]))
    
    p = Progress(n_turns, desc="Generating animation: ")
    
    @inbounds record(fig, filename, 1:n_turns; framerate=100) do frame_idx
        empty!(ax_z)
        empty!(ax_E)
        empty!(ax_phase)
        
        frame_data = particles_out[frame_idx]
        σ_E = std(frame_data.ΔE)
        σ_z = std(frame_data.z)
        ylims!(ax_phase, (minimum(boundary_points[2])/σ_E - 4,maximum(boundary_points[2])/σ_E + 4))
        scatter!(ax_phase, frame_data.ϕ, frame_data.ΔE/σ_E, color=:black, markersize=1)
        lines!(ax_phase, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
        
        z_data = frame_data.z
        E_data = frame_data.ΔE
        
        z_normalized = @view(z_data[.-5 .< z_data ./ σ_z .< 5]) ./ σ_z
        E_normalized = @view(E_data[.-5 .< E_data ./ σ_E .< 5]) ./ σ_E
        
        z_density = kde(z_normalized, boundary=(-5,5), npoints=200)
        E_density = kde(E_normalized, boundary=(-5,5), npoints=200)

        
        
        lines!(ax_z, z_density.x, z_density.density, color=:red, linewidth=2)
        lines!(ax_E, E_density.x, E_density.density, color=:red, linewidth=2)
        
        title_label.text = "Turn $frame_idx"
        next!(p)
    end
    println("Animation complete!")
end

"""
    animate_one_by_one(n_turns, particle_states, ϕs, α_c, mass, voltage, harmonic, E0, σ_E, σ_z, 
                      acc_radius, freq_rf, pipe_radius; frame_itv=1, filename="anim_1_by_1.mp4") -> Nothing

Generate animation of beam evolution with configurable frame intervals, evolving particles between frames.

# Arguments
- `n_turns::Int64`: Total number of turns
- `particle_states::StructArray{ParticleState{Float64}}`: Initial particle states
- `ϕs::Float64`: Synchronous phase
- `α_c::Float64`: Momentum compaction factor
- `mass::Float64`: Particle mass
- `voltage::Float64`: RF voltage
- `harmonic::Int64`: RF harmonic number
- `E0::Float64`: Reference energy
- `σ_E::Float64`: Energy spread
- `σ_z::Float64`: Bunch length
- `acc_radius::Float64`: Accelerator radius
- `freq_rf::Float64`: RF frequency
- `pipe_radius::Float64`: Beam pipe radius

# Keywords
- `frame_itv::Int=1`: Number of turns between frames
- `filename::String="anim_1_by_1.mp4"`: Output filename

# Implementation Notes
- Evolves particles between animation frames
- Includes wakefield effects
- Updates phase space and distribution plots
- Supports variable frame intervals
- Creates timestamped output directories

# Example
```julia
animate_one_by_one(1000, initial_states, 0.0, 1e-3, mass_electron, 1e6, 400, 3e9, 
                  1e-4, 1e-3, 100.0, 500e6, 0.02, frame_itv=10)
```
"""
function animate_one_by_one(
    n_turns::Int64, particle_states::StructArray{ParticleState{Float64}},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64, acc_radius::Float64,
    freq_rf::Float64, pipe_radius::Float64,; frame_itv::Int=1,
    filename::String="anim_1_by_1.mp4")

    # Directory setup
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particle_states))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    filename = joinpath(folder_storage, filename)

    # Physics calculations
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    
    # Pre-compute densities
    # z_densities, E_densities = precompute_densities(particles_out, σ_z, σ_E)
    
    # Setup figure
    println("Setting up figure...")
    fig = Figure(;size=(1400, 900), font="Arial")
    ax_z = Axis(fig[1, 1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
    ax_E = Axis(fig[1, 2], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
    ax_phase = Axis(fig[2, 1:2], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
    title_label = Label(fig[0, :], "Turn 1", fontsize=20, halign=:center)
    
    # Set static limits
    ylims!(ax_z, (0, 0.5))
    ylims!(ax_E, (0, .5))
    xlims!(ax_z, (-5, 5))
    xlims!(ax_E, (-5, 5))
    xlims!(ax_phase, (0, 3π/2))

    σ_E, σ_z, E0 =  longitudinal_evolve!(
            frame_itv, particle_states, ϕs, α_c, mass, voltage,
            harmonic, acc_radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
            use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
            use_excitation=true, display_counter=false)
    
    
    boundary_obs = Observable((boundary_points[1], boundary_points[2]))
    
    p = Progress(n_turns, desc="Generating animation: ")
    
    @inbounds record(fig, filename, 1:n_turns; framerate=100) do frame_idx
        empty!(ax_z)
        empty!(ax_E)
        empty!(ax_phase)
        
        frame_data = particle_states
        
        ylims!(ax_phase, (minimum(boundary_points[2])/σ_E - 4,maximum(boundary_points[2])/σ_E + 4))
        scatter!(ax_phase, frame_data.ϕ, frame_data.ΔE/σ_E, color=:black, markersize=1)
        lines!(ax_phase, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
        
        z_data = frame_data.z
        E_data = frame_data.ΔE
        
        z_normalized = @view(z_data[.-5 .< z_data ./ σ_z .< 5]) ./ σ_z
        E_normalized = @view(E_data[.-5 .< E_data ./ σ_E .< 5]) ./ σ_E
        
        z_density = kde(z_normalized, boundary=(-5,5), npoints=200)
        E_density = kde(E_normalized, boundary=(-5,5), npoints=200)

        
        
        lines!(ax_z, z_density.x, z_density.density, color=:red, linewidth=2)
        lines!(ax_E, E_density.x, E_density.density, color=:red, linewidth=2)
        
        title_label.text = "Turn $frame_idx"
        
        σ_E, σ_z, E0 =  longitudinal_evolve!(
            frame_itv, particle_states, ϕs, α_c, mass, voltage,
            harmonic, acc_radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
            use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
            use_excitation=true, display_counter=false)
            
        next!(p)
    end
    println("Animation complete!")
end

"""
    calculate_histogram(data::Vector{Float64}, bins::Int64) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate histogram of particle distribution with optimized binning.

# Arguments
- `data::Vector{Float64}`: Input data vector
- `bins::Int64`: Number of histogram bins

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Bin centers and counts

# Implementation Notes
- Uses FHist for efficient histogram computation
- Optimized for large datasets
- Returns centers and counts for plotting

# Example
```julia
centers, counts = calculate_histogram(particle_positions, 100)
```
"""
@inline function calculate_histogram(data::Vector{Float64}, bins_edges)
    histo = Hist1D(data, binedges=bins_edges)
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end

"""
    calculate_kde(data::Vector{Float64}, bandwidth=nothing) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate kernel density estimate of particle distribution with automatic bandwidth selection.

# Arguments
- `data::Vector{Float64}`: Input data vector
- `bandwidth::Union{Float64, Nothing}=nothing`: Optional bandwidth parameter

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: x-coordinates and density values

# Implementation Notes
- Uses Silverman's rule of thumb for automatic bandwidth: 1.06 * σ * n^(-1/5)
- Optimized for smooth distribution estimation
- Handles edge cases appropriately

# Example
```julia
x_coords, density = calculate_kde(particle_positions)
x_coords, density = calculate_kde(particle_positions, bandwidth=0.1)
```
"""
@inline function calculate_kde(data::Vector{Float64}, bandwidth=nothing)
    if isnothing(bandwidth)
        bandwidth = 1.06 * std(data) * length(data)^(-0.2)
    end
    kde_obj = kde(data; bandwidth=bandwidth)
    return kde_obj.x, kde_obj.density
end

"""
    create_simulation_buffers(n_particles::Int, nbins::Int, T::Type=Float64) -> SimulationBuffers{T}

Create pre-allocated buffers for efficient simulation calculations.

# Arguments
- `n_particles::Int`: Number of particles in simulation
- `nbins::Int`: Number of bins for histogram calculations
- `T::Type=Float64`: Numeric type for calculations

# Returns
- `SimulationBuffers{T}`: Pre-allocated buffers for simulation

# Implementation Notes
- Allocates memory once at initialization
- Supports both particle and histogram calculations
- Includes buffers for wakefield calculations
- Optimized for SIMD operations

# Example
```julia
buffers = create_simulation_buffers(10000, 100)
```
"""
function create_simulation_buffers(n_particles::Int, nbins::Int, T::Type=Float64)
    SimulationBuffers{T}(
        Vector{T}(undef, n_particles),  # WF
        Vector{T}(undef, n_particles),  # potential
        Vector{T}(undef, n_particles),  # Δγ
        Vector{T}(undef, n_particles),  # η
        Vector{T}(undef, n_particles),  # coeff
        Vector{T}(undef, n_particles),  # temp_z
        Vector{T}(undef, n_particles),  # temp_ΔE
        Vector{T}(undef, n_particles),  # temp_ϕ
        Vector{T}(undef, nbins),       # WF_temp
        Vector{T}(undef, nbins),       # λ
        Vector{Complex{T}}(undef, nbins) # convol
    )
end



"""
    pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T -> Nothing

Pad vectors to power-of-two length for efficient FFT operations.

# Arguments
- `f::AbstractVector{T}`: First vector to pad
- `g::AbstractVector{T}`: Second vector to pad
- `power_two_length::Int`: Target length (must be power of 2)

# Implementation Notes
- Modifies vectors in-place to save memory
- Ensures lengths are suitable for FFT
- Zero-pads the extended regions
- Preserves original data

# Example
```julia
pad_and_ensure_power_of_two!(signal1, signal2, 1024)
```
"""
function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T
    N = length(f)
    M = length(g)
    
    original_f = copy(f)
    resize!(f, power_two_length)
    f[1:N] = original_f
    f[N+1:end] .= zero(T)
    
    original_g = copy(g)
    resize!(g, power_two_length)
    g[1:M] = original_g
    g[M+1:end] .= zero(T)
    
    return nothing
end

"""
    make_separatrix(ϕs::Float64, voltage::Float64, energy::Float64, 
                   harmonic::Int64, η::Float64, β::Float64) 
                   -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate separatrix coordinates for phase space visualization using optimized numerical methods.

# Arguments
- `ϕs::Float64`: Synchronous phase
- `voltage::Float64`: RF voltage
- `energy::Float64`: Reference energy
- `harmonic::Int64`: RF harmonic number
- `η::Float64`: Slip factor
- `β::Float64`: Relativistic beta

# Returns
- Tuple of vectors containing phase and energy coordinates for separatrix

# Example
```julia
# Calculate separatrix for given parameters
phases, energies = make_separatrix(0.0, 1e6, 1e9, 400, 1e-3, 0.999999)
```
"""
function make_separatrix(ϕs::Float64, voltage::Float64, energy::Float64, 
    harmonic::Int64, η::Float64, β::Float64)
    # Pre-calculate constants to avoid repeated computation
    const_factor = voltage * energy * β^2 / (harmonic * π * η)

    # Improved root finding function with better numerical stability
    function fangle(ϕu)
        Δϕ = π - ϕu - ϕs
        return -cos(ϕu) - cos(ϕs) + sin(ϕs) * Δϕ
    end

    # More robust initial bracket for root finding
    ϕ_lower = max(-2π, ϕs - 2π)
    ϕ_upper = min(2π, ϕs + 2π)

    # Use more robust root finding method
    ϕ_unstable = find_zero(fangle, (ϕ_lower, ϕ_upper), Roots.Brent())

    # Optimize the number of points based on the region of interest
    Δϕ = π - ϕs - ϕ_unstable
    n_points = max(1000, round(Int, abs(Δϕ) * 500))  # Scale points with separatrix size

    # Use LinRange for more efficient memory allocation
    ϕtest = LinRange(ϕ_unstable, π-ϕs, n_points)

    # Preallocate arrays
    sep = Vector{Float64}(undef, n_points)

    # Vectorize the main calculation
    @. sep = sqrt(abs(const_factor * (cos(ϕtest) + cos(ϕs) - sin(ϕs) * (π - ϕtest - ϕs))))

    # Create the full separatrix more efficiently
    sep_total = Vector{Float64}(undef, 2n_points)
    ϕ_test_total = Vector{Float64}(undef, 2n_points)

    # Fill both halves simultaneously
    @views begin
        sep_total[1:n_points] = reverse(sep)
        sep_total[n_points+1:end] = -sep
        ϕ_test_total[1:n_points] = reverse(ϕtest)
        ϕ_test_total[n_points+1:end] = ϕtest
    end

    return (ϕ_test_total, sep_total)
end ;


"""
    histogram_particle_data(particles_out::BeamTurn, turn_number::Int64; 
                          z_hist::Bool=true, e_hist::Bool=true, 
                          filename::String="histogram_particle.png", 
                          save_figs::Bool=true) -> Union{Figure, Tuple{Figure, Figure}, Nothing}

Generate histograms of particle distributions at a specific turn.

# Arguments
- `particles_out::BeamTurn`: Beam trajectory data
- `turn_number::Int64`: Turn number to analyze

# Keywords
- `z_hist::Bool=true`: Generate longitudinal distribution histogram
- `e_hist::Bool=true`: Generate energy distribution histogram
- `filename::String="histogram_particle.png"`: Output filename
- `save_figs::Bool=true`: Save figures to files

# Returns
- Single Figure if only one histogram requested
- Tuple of Figures if both histograms requested
- Nothing if no histograms requested

# Implementation Notes
- Uses KDE for smooth distribution estimation
- Applies appropriate normalization
- Creates timestamped output directories
- Supports selective histogram generation

# Example
```julia
# Generate both histograms
z_fig, e_fig = histogram_particle_data(beam_data, 100)

# Generate only energy histogram
e_fig = histogram_particle_data(beam_data, 100, z_hist=false)
```
"""
function histogram_particle_data(particles_out::BeamTurn, turn_number::Int64, ;
                               z_hist::Bool=true, e_hist::Bool=true,
                               filename::String="histogram_particle.png",
                               save_figs::Bool=true)
    
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    
    n_bins = 10^(ceil(Int, log10(length(particles_out[1])))-2)

    if z_hist
        z_bin_width = (5 - (-5)) / n_bins
        z_bins = range(-5, 5, step=z_bin_width)
        z_normalized = particles_out[turn_number].z / σ_z
        z_centers, z_counts = calculate_fixed_histogram(z_normalized, z_bins)
        z_kde_x, z_kde_y = calculate_kde(z_normalized)
        
        fig_z = Figure()
        ax_z = Axis(fig_z[1,1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
        barplot!(ax_z, z_centers, z_counts, color=(:red, 0.5))
        lines!(ax_z, z_kde_x, z_kde_y .* length(z_normalized) .* z_bin_width,
               color=:green, linewidth=2)
        
        if save_figs
            filename = joinpath(folder_storage, filename)
            if z_hist && e_hist
                filename = "histogram_particle_z.png"
                filename = joinpath(folder_storage, filename)
            end
            save(filename)
        end
    end
    
    if e_hist
        E_bin_width = (120 - (-120)) / n_bins
        E_bins = range(-120, 120, step=E_bin_width)
        E_normalized = particles_out[turn_number].ΔE / σ_E
        E_centers, E_counts = calculate_fixed_histogram(E_normalized, E_bins)
        E_kde_x, E_kde_y = calculate_kde(E_normalized)

        fig_E = Figure()
        ax_E = Axis(fig_E[1,1], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
        barplot!(ax_E, E_centers, E_counts, color=(:red, 0.5))
        lines!(ax_E, E_kde_x, E_kde_y .* length(E_normalized) .* E_bin_width,
               color=:green, linewidth=2)
        
        if save_figs
            filename = joinpath(folder_storage, filename)
            if z_hist && e_hist
                filename = "histogram_particle_E.png"
                filename = joinpath(folder_storage, filename)
            end
            save(filename)
        end
    end

    if z_hist && e_hist
        return fig_z, fig_E
    elseif z_hist
        return fig_z
    elseif e_hist
        return fig_E
    else
        println("No histogram selected")
        return nothing
    end
end

"""
    scatter_particle_data(particle_states::BeamTurn{Float64}, turn_number::Int64,
                         ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
                         harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64;
                         filename::String="particle_scatter.png",
                         ϕ_plot::Bool=true,
                         save_fig::Bool=true) -> Figure

Generate scatter plots of particle distributions with phase space or position-energy coordinates.

# Arguments
- `particle_states::BeamTurn{Float64}`: Beam trajectory data
- `turn_number::Int64`: Turn number to plot
- `ϕs::Float64`: Synchronous phase
- `α_c::Float64`: Momentum compaction factor
- `mass::Float64`: Particle mass
- `voltage::Float64`: RF voltage
- `harmonic::Int64`: RF harmonic number
- `E0::Float64`: Reference energy
- `σ_E::Float64`: Energy spread
- `σ_z::Float64`: Bunch length

# Keywords
- `filename::String="particle_scatter.png"`: Output filename
- `ϕ_plot::Bool=true`: Plot phase space (true) or position-energy (false)
- `save_fig::Bool=true`: Save figure to file

# Returns
- `Figure`: Makie figure object

# Implementation Notes
- Supports both phase space and position-energy plotting
- Includes separatrix calculation for phase space plots
- Automatically handles axis scaling and labels
- Creates timestamped output directories

# Example
```julia
# Phase space plot
fig = scatter_particle_data(beam_data, 100, 0.0, 1e-3, mass_electron,
                          1e6, 400, 3e9, 1e-4, 1e-3)

# Position-energy plot
fig = scatter_particle_data(beam_data, 100, 0.0, 1e-3, mass_electron,
                          1e6, 400, 3e9, 1e-4, 1e-3, ϕ_plot=false)
```
"""
function scatter_particle_data(particle_states::BeamTurn{Float64}, turn_number::Int64,
                             ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
                             harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64,;
                             filename::String="particle_scatter.png",
                             ϕ_plot::Bool=true,
                             save_fig::Bool=true)
    
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)

    z_data = particle_states[turn_number].z
    ϕ_data = particle_states[turn_number].ϕ
    E_data = particle_states[turn_number].ΔE
    
    fig = Figure(size=(800, 500))
    
    if ϕ_plot
        ax = Axis(fig[1,1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        scatter!(ax, ϕ_data, E_data/σ_E, color=:black, markersize=1)
        lines!(ax, boundary_points[1], boundary_points[2]/σ_E, color=:red)
    else
        ax = Axis(fig[1,1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        scatter!(ax, z_data/σ_z, E_data/σ_E, color=:black, markersize=1)
    end
    
    if save_fig
        save(filename)
    end
    
    return fig
end

"""
    BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T -> BeamTurn{T,N}

Constructor for BeamTurn object to store particle states across multiple turns.

# Arguments
- `n_turns::Integer`: Number of turns to simulate
- `n_particles::Integer`: Number of particles

# Returns
- `BeamTurn{T,N}`: BeamTurn object with pre-allocated storage

# Example
```julia
beam = BeamTurn{Float64}(1000, 10000)  # For 1000 turns, 10000 particles
```
"""
function BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T
    states = SVector{n_turns+1}(
        StructArray{ParticleState{T}}((
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles)
        )) for _ in 1:n_turns+1
    )
    return BeamTurn{T, n_turns+1}(states)
end

"""
    copyto!(dest::BeamTurn, turn_idx::Integer, src::StructArray{ParticleState{T}}) where T -> BeamTurn

Copy particle states to a specific turn in the BeamTurn object.

# Arguments
- `dest::BeamTurn`: Destination BeamTurn object
- `turn_idx::Integer`: Turn index to copy to
- `src::StructArray{ParticleState{T}}`: Source particle states

# Returns
- `BeamTurn`: The destination object

# Example
```julia
copyto!(trajectory, 10, current_states)  # Copy to turn 10
```
"""
function Base.copyto!(dest::BeamTurn, turn_idx::Integer, src::StructArray{ParticleState{T}}) where T
    copyto!(dest.states[turn_idx].z, src.z)
    copyto!(dest.states[turn_idx].ΔE, src.ΔE)
    copyto!(dest.states[turn_idx].ϕ, src.ϕ)
    return dest
end

"""
    copyto!(dest::BeamTurn, turn_idx::Integer, x::AbstractVector, px::AbstractVector, z::AbstractVector) -> BeamTurn

Copy coordinate arrays to a specific turn in the BeamTurn object.

# Arguments
- `dest::BeamTurn`: Destination BeamTurn object
- `turn_idx::Integer`: Turn index to copy to
- `x::AbstractVector`: Position coordinates
- `px::AbstractVector`: Momentum coordinates
- `z::AbstractVector`: Longitudinal coordinates

# Returns
- `BeamTurn`: The destination object

# Example
```julia
copyto!(trajectory, 5, positions, momenta, longitudinal)
```
"""
function Base.copyto!(dest::BeamTurn, turn_idx::Integer, x::AbstractVector, px::AbstractVector, z::AbstractVector)
    copyto!(dest.states[turn_idx].x, x)
    copyto!(dest.states[turn_idx].px, px)
    copyto!(dest.states[turn_idx].z, z)
    return dest
end

"""
    calculate_fixed_histogram(data::Vector{Float64}, bins::AbstractRange) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate histogram with fixed bin edges for consistent visualization.

# Arguments
- `data::Vector{Float64}`: Input data vector
- `bins::AbstractRange`: Pre-defined bin edges

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Bin centers and counts

# Implementation Notes
- Uses fixed bins for consistent visualization
- Returns centers for plotting
- Optimized for visualization purposes

# Example
```julia
bins = range(-5, 5, length=100)
centers, counts = calculate_fixed_histogram(particle_positions, bins)
```
"""
function calculate_fixed_histogram(data::Vector{Float64}, bins::AbstractRange)
    hist = fit(Histogram, data, bins)
    centers = (bins[1:end-1] + bins[2:end]) ./ 2
    return centers, hist.weights
end

"""
    make_separatrix_extended(ϕs::Float64, voltage::Float64, energy::Float64,
                           harmonic::Int64, η::Float64, β::Float64) 
                           -> Tuple{Vector{Float64}, Vector{Float64}, Float64, Float64}

Calculate separatrix coordinates and bucket parameters.

# Arguments
- `ϕs::Float64`: Synchronous phase
- `voltage::Float64`: RF voltage
- `energy::Float64`: Reference energy
- `harmonic::Int64`: RF harmonic number
- `η::Float64`: Slip factor
- `β::Float64`: Relativistic beta

# Returns
- `Tuple` containing:
  1. Phase coordinates (Vector{Float64})
  2. Energy coordinates (Vector{Float64})
  3. Bucket height (Float64)
  4. Bucket area (Float64)

# Implementation Notes
- Extends basic separatrix calculation
- Includes bucket parameter calculations
- Uses Hamiltonian formalism
- Optimized for accuracy at bucket edges

# Example
```julia
ϕ, ΔE, height, area = make_separatrix_extended(0.0, 1e6, 3e9, 400, 1e-3, 0.999999)
```
"""
function make_separatrix_extended(ϕs::Float64, voltage::Float64, energy::Float64, 
                                harmonic::Int64, η::Float64, β::Float64)
    # Calculate basic separatrix
    ϕ_total, sep_total = make_separatrix(ϕs, voltage, energy, harmonic, η, β)

    # Calculate bucket parameters
    const_factor = voltage * energy * β^2 / (harmonic * π * η)
    bucket_height = sqrt(abs(const_factor * (2)))  # Maximum height
    bucket_width = π - ϕs
    bucket_area = 4 * bucket_height * bucket_width  # Approximate area

    return (ϕ_total, sep_total, bucket_height, bucket_area)
end

"""
    Base.getindex(pt::BeamTurn, i::Integer) -> StructArray{ParticleState{T}}

Get particle states for a specific turn from a BeamTurn object.

# Arguments
- `pt::BeamTurn`: BeamTurn object
- `i::Integer`: Turn index

# Returns
- `StructArray{ParticleState{T}}`: Particle states at requested turn

# Example
```julia
states_at_turn_10 = beam_trajectory[10]
```
"""
Base.getindex(pt::BeamTurn, i::Integer) = pt.states[i]

"""
    Base.iterate(pt::BeamTurn, state=1) -> Union{Tuple{StructArray{ParticleState{T}}, Int}, Nothing}

Iterator interface for BeamTurn object to enable iteration over turns.

# Arguments
- `pt::BeamTurn`: BeamTurn object
- `state=1`: Current iteration state

# Returns
- `Nothing` if iteration complete
- `Tuple` of (particle_states, next_state) otherwise

# Example
```julia
for turn_states in beam_trajectory
    # Process each turn's states
end
```
"""
Base.iterate(pt::BeamTurn, state=1) = state > length(pt.states) ? nothing : (pt.states[state], state + 1)

"""
    Base.length(pt::BeamTurn{T,N}) where {T,N} -> Int

Get number of turns stored in BeamTurn object.

# Arguments
- `pt::BeamTurn{T,N}`: BeamTurn object

# Returns
- `Int`: Number of turns (N)

# Example
```julia
n_turns = length(beam_trajectory)
```
"""
Base.length(pt::BeamTurn{T,N}) where {T,N} = N

"""
    fast_reset_buffers!(buffers::SimulationBuffers{T}) where T -> Nothing

Optimized version of buffer reset using SIMD operations and batch processing.
More efficient for large arrays but with same functionality as reset_buffers!.

# Arguments
- `buffers::SimulationBuffers{T}`: Pre-allocated simulation buffers

# Implementation Notes
- Uses @turbo for SIMD optimization
- Processes buffers in chunks for cache efficiency
- No conditional checks (always resets)
- More efficient for large arrays
"""
function fast_reset_buffers!(buffers::SimulationBuffers{T}) where T
    @turbo for i in eachindex(buffers.WF)
        buffers.WF[i] = zero(T)
        buffers.potential[i] = zero(T)
        buffers.Δγ[i] = zero(T)
        buffers.η[i] = zero(T)
        buffers.coeff[i] = zero(T)
        buffers.temp_z[i] = zero(T)
        buffers.temp_ΔE[i] = zero(T)
        buffers.temp_ϕ[i] = zero(T)
    end

    @turbo for i in eachindex(buffers.WF_temp)
        buffers.WF_temp[i] = zero(T)
        buffers.λ[i] = zero(T)
        buffers.convol[i] = zero(Complex{T})
    end

    return nothing
end

"""
    reset_specific_buffers!(buffers::SimulationBuffers{T}, buffer_names::Vector{Symbol}) where T -> Nothing

Reset only specified buffers by name.

# Arguments
- `buffers::SimulationBuffers{T}`: Pre-allocated simulation buffers
- `buffer_names::Vector{Symbol}`: Names of buffers to reset

# Example
```julia
# Reset only WF and potential buffers
reset_specific_buffers!(sim_buffers, [:WF, :potential])
```
"""
function reset_specific_buffers!(buffers::SimulationBuffers{T}, buffer_names::Vector{Symbol}) where T
    for name in buffer_names
        buffer = getfield(buffers, name)
        if !all(iszero, buffer)
            fill!(buffer, isa(eltype(buffer), Complex) ? zero(Complex{T}) : zero(T))
        end
    end
    return nothing
end





##############################################################################################################


E0_ini = 4e9 ;
mass = MASS_ELECTRON ;
voltage = 5e6 ;
harmonic = 360 ;
radius = 250. ;
pipe_radius = .00025 ;

α_c = 3.68e-4 ;
γ = E0_ini/mass ;
β = sqrt(1 - 1/γ^2) ;
η = α_c - 1/γ^2 ;
sin_ϕs = 0.5 ;
ϕs = 5π/6 ;
freq_rf = 180.15e7 ;


μ_E = 0. ;
μ_z = 0. ;
ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT)) ;
# σ_E0 = 1.938e6
σ_E0 = 1e6 ;
σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini ;



# ν_s = sqrt(voltage*harmonic*η*abs(cos(ϕs))/(2*π*energy*β^2))
# ν_s / (harmonic * abs(η))
### WHEN YOU WANT TO SEE POTENTIAL AND WF PLOTS FOR SANITY CHECKS
n_turns = 1000;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),E0_ini,mass,ϕs, freq_rf) ;
particles_out, plot_potential, plot_WF= longitudinal_evolve(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E0, σ_z0,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=true,plot_WF=true, write_to_file=false, output_file="test1.h5") ;  
plot_potential[1]
plot_potential[10]
plot_potential[50]
plot_potential[1000]
plot_WF[10]
scatter(particles_out[150].ϕ, particles_out[150].ΔE / σ_E, markersize=1, color=:black)


### Getting video with histrograms, one frame at a time
n_turns = 20000;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),E0_ini,mass,ϕs, freq_rf) ;
particles_out= longitudinal_evolve(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E0, σ_z0,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5") ;  
all_animate_optimized(length(particles_out[1]), particles_out[1], ϕs, α_c, mass, voltage, harmonic,E0_ini, σ_E0, σ_z0, "opt_anim_0_1.mp4") ;
particles_out[end]

### Getting video with histrograms, multiple frames at a time
n_turns = 100;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),E0_ini,mass,ϕs, freq_rf) ;
animate_one_by_one(n_turns, particle_states, ϕs, α_c, mass, voltage, harmonic,E0, σ_E, σ_z, 
    radius, freq_rf, pipe_radius, frame_itv=1, filename = "anim_1_by_1.mp4")

### To do inplace evolution of particles
n_turns = 50000;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e6),E0_ini,mass,ϕs, freq_rf) ;
σ_E, σ_z, E0 = longitudinal_evolve!(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true);

println("σ_E = ", σ_E, " E0 = ", E0)
scatter(particle_states.ϕ, particle_states.ΔE / σ_E, markersize=1, color=:black)
# fig = Figure(;size = (800, 500))
# ax = Axis(fig[1,1], xlabel=L"z / \sigma_z", ylabel="Count", title = "z Distribution")
hist( particle_states.ΔE / σ_E, bins=100)

hist(particle_states.z / σ_z, bins=100)

# display(fig)
# save("Haissinski/pres_plots/z_hist_final.png", fig)

# fig = Figure(;size = (800, 500))
# ax = Axis(fig[1,1], xlabel=L"Turn", ylabel=L"\sigma _E", title = L"\sigma _E Evolution", xscale = log10)
# lines!(ax, collect(range(1, length(temp_vec))), temp_vec)
# # xlims!(ax, 10000, n_turns)
# display(fig)
# save("Haissinski/pres_plots/energy_std_evo_goes_just_right.png", fig)

resize!(particles_out.states, 500)[1]

### To get scatter plot animation only, but faster than the other methods
n_turns = 100;
particle_states, σ_E, σ_z = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),energy,mass,ϕs, freq_rf) ;
scatter_plots = longitudinal_evolve!(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_scatter=true);
create_animation_from_pngs(scatter_plots, 100000, filename="scatter_fast_test.mp4")


### To get all plots, quickly
n_turns = 1;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),E0_ini,mass,ϕs, freq_rf) ;
scatter_plots, potential_plots = longitudinal_evolve!(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=false, plot_scatter=true, plot_potential=true, plot_WF=false);
scatter_plots[1]
# save("Haissinski/pres_plots/phase_plot_ini_1e5.png", scatter_plots[1])
scatter_plots[2]
potential_plots[2]
# save("Haissinski/pres_plots/wake_potential_ini_1e5.png", potential_plots[1])
# scatter(particle_states.ϕ, particle_states.ΔE , markersize=1, color=:black)
# hist(particle_states.ΔE , bins=100)

fig = Figure(;size = (800, 500))
ax = Axis(fig[1,1], xlabel=L"\Delta E / \sigma_E", ylabel="Count", title = "Energy Distribution") 
hist!(ax, particle_states.ΔE / σ_E, bins=100)
display(fig)
save("Haissinski/pres_plots/E_dist_1e6.png", fig)

fig = Figure(;size = (800, 500))
ax = Axis(fig[1,1], xlabel=L"\Delta z / \sigma_z", ylabel="Count", title = "z Distribution") 
hist!(ax, particle_states.z / σ_z, bins=100)
display(fig)
save("Haissinski/pres_plots/z_dist_1e6.png", fig)

############ BENCHMARKING ############
n_turns = 1000;
particle_states, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e5),E0_ini,mass,ϕs, freq_rf) ;
@btime longitudinal_evolve(
    $n_turns, $particle_states, $ϕs, $α_c, $mass, $voltage,
    $harmonic, $radius, $freq_rf, $pipe_radius, $energy, $σ_E,$σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5");
    # 189.123 ms (20246 allocations: 275.54 MiB), 1e5 particles, 1e2 turns
    # 3.828 s (22982 allocations: 2.87 GiB), 1e6 particles, 1e2 turns
    # 2.989 s (208017 allocations: 2.61 GiB), 1e5 particles, 1e3 turns
    # 33.730 s (236678 allocations: 25.47 GiB), 1e6 particles, 1e3 turns NEEDS TO BE UPDATED, LIKELY LOWER

@btime longitudinal_evolve!(
    $n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0_ini, σ_E0,σ_z0,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true) ; 
    #150.672 ms (9300 allocations: 44.14 MiB), 1e5 particles, 1e2 turns
    #1.861 s (11613 allocations: 630.50 MiB), 1e6 particles, 1e2 turns <0.5% GC
    #24.179 s (20394 allocations: 4.85 GiB), 1e7 particles, 1e2 turns
    #1.496 s (98204 allocations: 383.75 MiB), 1e5 particles, 1e3 turns <0.5% GC
    #18.682 s (122755 allocations: 5.59 GiB), 1e6 particles, 1e3 turns 
    #206.572 s (182878 allocations: 42.87 GiB), 1e7 particles, 1e3 turns

benchmark_result =  @benchmark longitudinal_evolve!(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true)

@btime generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e7),energy,mass,ϕs, freq_rf) ; 
# 14.387 ms (200030 allocations: 12.06 MiB), 1e5 particles
# 147.504 ms (2000030 allocations: 115.05 MiB), 1e6 particles
# 1.811 s (20000030 allocations: 1.12 GiB), 1e7 particles

function benchmark_filter_outliers(trial::BenchmarkTools.Trial; percentile=0.90)
    cutoff = quantile(trial.times, percentile)
    keep = trial.times .<= cutoff
    
    filtered = deepcopy(trial)
    filtered.times = trial.times[keep]
    filtered.gctimes = trial.gctimes[keep]
    
    return filtered
end

benchmark_result = @benchmark longitudinal_evolve(
        $n_turns, $particle_states, $ϕs, $α_c, $mass, $voltage,
        $harmonic, $radius, $freq_rf, $pipe_radius, $energy, $σ_E,$σ_z,
        use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
        use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5")

filtered_result = benchmark_filter_outliers(benchmark_result)


@ProfileSVG.profview longitudinal_evolve(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5") 



nbins1 = next_power_of_two(Int(10^(ceil(Int, log10(length(particle_states.z))-2))))
bin_edges1 = range(-7.5*σ_z, 7.5*σ_z, length=nbins1+1)

@btime calculate_histogram(particle_states.z, bin_edges1) ;
@btime calculate_kde(particle_states.z) ;
@btime hist(particle_states.z, bin_edges1 ) ;