# """
# evolution.jl - Particle evolution and beam dynamics

# This file implements the core longitudinal evolution algorithm and particle generation.
# It handles the multi-turn tracking of particles through the accelerator,
# including RF cavity effects, synchrotron radiation, and collective effects.
# """

# using StructArrays
# using StaticArrays
# using Statistics
# using Random
# using Distributions
# using ProgressMeter
# using LoopVectorization

# """
#     generate_particles(
#         μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
#         energy::T, mass::T, ϕs::T, freq_rf::T
#     ) where T<:Float64 -> Tuple{StructArray{Particle{T}}, T, T, T}

# Generate initial particle distribution.
# """
# function generate_particles(
#     μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
#     energy::T, mass::T, ϕs::T, freq_rf::T) where T<:Float64

#     # Initial sampling for covariance estimation
#     initial_sample_size::Int = min(10_000, num_particles)
#     z_samples = rand(Normal(μ_z, σ_z), initial_sample_size)
#     E_samples = rand(Normal(μ_E, σ_E), initial_sample_size)

#     # Compute covariance matrix
#     Σ = Symmetric([cov(z_samples, z_samples) cov(z_samples, E_samples);
#                    cov(z_samples, E_samples) cov(E_samples, E_samples)])

#     # Create multivariate normal distribution
#     μ = SVector{2,T}(μ_z, μ_E)
#     dist_total = MvNormal(μ, Σ)

#     # Relativistic factors
#     γ::T = energy / mass
#     β::T = sqrt(1 - 1/γ^2)
#     rf_factor::T = freq_rf * 2π / (β * SPEED_LIGHT)

#     # Generate correlated random samples
#     samples = rand(dist_total, num_particles)  # 2 × num_particles matrix
#     z_vals = samples[1, :]
#     ΔE_vals = samples[2, :]


#     # Create the StructArray of Particles

#     # particles = StructArray{Particle{Float64}}(StructArray(Coordinate.(z_vals, ΔE_vals)))


#     particles = StructArray{Particle{Float64}}((
#     StructArray(Coordinate.(z_vals, ΔE_vals)),  # coordinates
#     # StructArray(Coordinate.(zeros(num_particles), zeros(num_particles)))  # uncertainty
#     ))


#     return particles, σ_E, σ_z, energy
# end


# """
#     longitudinal_evolve!(
#         particles::StructArray{Particle{T}},
#         params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
#         buffers::SimulationBuffers{T}
#     ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF} -> Tuple{T, T, TE}

# Evolve particles through the accelerator for multiple turns.
# Type-stable implementation supporting StochasticTriple parameters.

# Returns: (σ_E, σ_z, E0) - Final energy spread, bunch length, and reference energy
# """
# # function longitudinal_evolve!(
# #     particles::StructArray{Particle{T}},
# #     params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
# #     buffers::SimulationBuffers{T}
# #     ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}
    
# #     # Extract parameters
# #     E0 = params.E0
# #     mass = params.mass
# #     voltage = params.voltage
# #     harmonic = params.harmonic
# #     radius = params.radius
# #     freq_rf = params.freq_rf
# #     pipe_radius = params.pipe_radius
# #     α_c = params.α_c
# #     ϕs = params.ϕs
# #     n_turns = params.n_turns
# #     use_wakefield = params.use_wakefield
# #     update_η = params.update_η
# #     update_E0 = params.update_E0
# #     SR_damping = params.SR_damping
# #     use_excitation = params.use_excitation
    
# #     # Pre-compute physical constants - handle StochasticTriple values
# #     # Use StochasticAD.propagate for all calculations that might involve StochasticTriple
# #     # γ0 = StochasticAD.propagate((energy, m) -> energy / m, E0, mass)
# #     # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
# #     # η0 = StochasticAD.propagate((alpha, gamma) -> alpha - 1/(gamma^2), α_c, γ0)
# #     γ0 = E0 / mass
# #     β0 = 1 - 1 / γ0 / γ0
# #     η0 = α_c -  1 / γ0 / γ0
# #     sin_ϕs = sin(ϕs)
# #     # rf_factor = StochasticAD.propagate((freq, beta) -> freq * 2π / (beta * SPEED_LIGHT), freq_rf, β0)
# #     rf_factor = calc_rf_factor(freq_rf, β0)
    
# #     # Get initial spreads
# #     n_particles::Int = length(particles)
# #     σ_E0::T = std(particles.coordinates.ΔE)
# #     σ_z0::T = std(particles.coordinates.z)
    
# #     if use_wakefield
# #         nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
# #         bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
# #         kp = convert(T, 3e1)
# #         Z0 = convert(T, 120π)
# #         cτ = convert(T, 4e-3)
        
# #         # For wake calculations, safely convert values using propagate
# #         # wake_factor_val = StochasticAD.propagate(r -> Z0 * SPEED_LIGHT / (π * r), pipe_radius)
# #         # wake_sqrt_val = StochasticAD.propagate(r -> sqrt(2 * kp / r), pipe_radius)

# #         wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
# #         wake_sqrt_val = sqrt(2 * kp / pipe_radius)
# #     end
    
# #     # Setup progress meter
# #     p = Progress(n_turns, desc="Simulating Turns: ")
    
# #     # Main evolution loop
# #     for turn in 1:n_turns
# #         # Calculate current spreads
# #         σ_E::T = std(particles.coordinates.ΔE)
# #         σ_z::T = std(particles.coordinates.z)
        
# #         # Store previous energy values for update_E0 if needed
# #         if update_E0
# #             ΔE_before = copy(particles.coordinates.ΔE)
# #         end
        
# #         # RF voltage kick
# #         rf_kick!(voltage, sin_ϕs, rf_factor, ϕs, particles)
        
# #         # Quantum excitation
# #         if use_excitation
# #             quantum_excitation!(E0, radius, σ_E0, buffers, particles)
# #         end
        
# #         # Synchrotron radiation damping
# #         if SR_damping
# #             synchrotron_radiation!(E0, radius, particles)
# #         end
        
# #         # Apply wakefield effects
# #         if use_wakefield
# #             # Use propagate for all calculations that might involve StochasticTriple
# #             # curr = StochasticAD.propagate(
# #             #     (eta, energy, rad) -> begin
# #             #         # Avoid explicit Float64 conversions
# #             #         particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
# #             #         return particle_factor / energy / (2*π*rad) * σ_z / (eta * σ_E0^2)
# #             #     end,
# #             #     η0, E0, radius
# #             # )
# #             particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
# #             curr = particle_factor / E0 / (2*π*radius) * σ_z / (η0 * σ_E0^2)
            
# #             apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, curr, σ_z, bin_edges)
            
# #             if update_E0
# #                 # Update reference energy based on collective effects
# #                 mean_ΔE_diff = mean(particles.coordinates.ΔE .- ΔE_before)
# #                 # E0 = StochasticAD.propagate((energy, diff) -> energy + diff, E0, mean_ΔE_diff)
# #                 E0 = E0 + mean_ΔE_diff

# #                 # Zero the mean energy deviation - FIXED VERSION
# #                 mean_ΔE = mean(particles.coordinates.ΔE)
# #                 # Use the safe update function instead of direct assignment
# #                 safe_update_energy!(particles, mean_ΔE)
# #             end
# #         end
        
# #         # Update reference energy if needed
# #         if update_E0
# #             # Use propagate for all operations that involve potential StochasticTriples
# #             # E0 = StochasticAD.propagate((energy, v, s) -> energy + v * s, E0, voltage, sin_ϕs)
# #             # γ0 = StochasticAD.propagate((energy, m) -> energy/m, E0, mass)
# #             # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)

# #             γ0 = E0 / mass
# #             β0 = 1 - 1 / γ0 / γ0
# #             η0 = α_c -  1 / γ0 / γ0
            
# #             # Adjust for radiation losses
# #             if SR_damping
# #                 # E0 = StochasticAD.propagate(
# #                 #     (energy, rad) -> begin
# #                 #         # Calculate radiation coefficient without explicit type conversion
# #                 #         radiation_coeff = 4 * 8.85e-5 * (energy/1e9)^3 / rad
# #                 #         # Apply energy loss
# #                 #         return energy - radiation_coeff * energy / 4
# #                 #     end,
# #                 #     E0, radius
# #                 # )

# #                 E0 = E0 - 4 * 8.85e-5 * (E0/1e9)^3 / radius * E0 / 4
# #                 # γ0 = StochasticAD.propagate((energy, m) -> energy/m, E0, mass)
# #                 # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
# #                 γ0 = E0 / mass
# #                 β0 = 1 - 1 / γ0 / γ0
# #             end
# #         end
        
# #         # Update phase advance
# #         if update_η
# #             for i in 1:n_particles
# #                 # Calculate slip factor for each particle using propagate
# #                 Δγ_i = particles.coordinates.ΔE[i] / mass
# #                 # η_i = StochasticAD.propagate(
# #                 #     (alpha, gamma, delta_gamma) -> alpha - 1/(gamma + delta_gamma)^2,
# #                 #     α_c, γ0, Δγ_i
# #                 # )
# #                 η_i = α_c - 1/(γ0 + Δγ_i)^2
# #                 # Use helper function for phase advance with proper StochasticTriple handling
# #                 # particles.coordinates.z[i] = StochasticAD.propagate(
# #                 #     (eta_i, h, beta, energy, rf, phi_s) -> begin
# #                 #         coeff_i = 2π * h * eta_i / (beta * beta * energy)
# #                 #         ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf, phi_s)
# #                 #         ϕ_i += coeff_i * particles.coordinates.ΔE[i]
# #                 #         return ϕ_to_z(ϕ_i, rf, phi_s)
# #                 #     end,
# #                 #     η_i, harmonic, β0, E0, rf_factor, ϕs
# #                 # )

# #                 coeff_i = 2π * harmonic * η_i / (β0 * β0 * E0)
# #                 ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
# #                 ϕ_i += coeff_i * particles.coordinates.ΔE[i]
# #                 particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)

# #             end
# #         else
# #             # Using constant slip factor - use the helper function
# #             apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
# #         end
        
# #         # Update RF factor with new beta
# #         # rf_factor = StochasticAD.propagate(
# #         #     (freq, beta) -> freq * 2π / (beta * SPEED_LIGHT),
# #         #     freq_rf, β0
# #         # )
# #         rf_factor = calc_rf_factor(freq_rf, β0)
# #         # Update progress
# #         next!(p)
# #     end
    
# #     # Final spreads
# #     σ_E = std(particles.coordinates.ΔE)
# #     σ_z = std(particles.coordinates.z)
    
# #     # Return the actual E0 type without conversion
# #     return σ_E, σ_z, E0
# # end

# function longitudinal_evolve!(
#     particles::StructArray{Particle{T}},
#     params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
#     buffers::SimulationBuffers{T};
#     show_progress::Bool = true
#     ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}
    
#     # Extract parameters
#     E0 = params.E0
#     mass = params.mass
#     voltage = params.voltage
#     harmonic = params.harmonic
#     radius = params.radius
#     freq_rf = params.freq_rf
#     pipe_radius = params.pipe_radius
#     α_c = params.α_c
#     ϕs = params.ϕs
#     n_turns = params.n_turns
#     use_wakefield = params.use_wakefield
#     update_η = params.update_η
#     update_E0 = params.update_E0
#     SR_damping = params.SR_damping
#     use_excitation = params.use_excitation
    
#     # Pre-compute physical constants - handle StochasticTriple values
#     γ0 = E0 / mass
#     β0 = 1 - 1 / γ0 / γ0
#     η0 = α_c -  1 / γ0 / γ0
#     sin_ϕs = sin(ϕs)
#     rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
#     # Get initial spreads
#     n_particles::Int = length(particles)
    
#     # Pre-compute standard deviations once before main loop
#     σ_E0::T = compute_std(particles.coordinates.ΔE)
#     σ_z0::T = compute_std(particles.coordinates.z)
    
#     # Initialize wake parameters
#     if use_wakefield
#         nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
#         bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
#         kp = convert(T, 3e1)
#         Z0 = convert(T, 120π)
#         cτ = convert(T, 4e-3)
        
#         wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
#         wake_sqrt_val = sqrt(2 * kp / pipe_radius)
#     end
    
#     # Setup progress meter
#     if show_progress
#         p = Progress(n_turns, desc="Simulating Turns: ")
#     end
    
#     # Pre-allocate energy buffer for updates
#     ΔE_before = Vector{T}(undef, n_particles)
    
#     # Main evolution loop
#     for turn in 1:n_turns
#         # Calculate current spreads - inline calculation to avoid temp arrays
#         σ_E::T = compute_std(particles.coordinates.ΔE) 
#         σ_z::T = compute_std(particles.coordinates.z)
        
#         # Store previous energy values for update_E0 if needed
#         if update_E0
#             copyto!(ΔE_before, particles.coordinates.ΔE)
#         end
        
#         # RF voltage kick
#         rf_kick!(voltage, sin_ϕs, rf_factor, ϕs, particles)
        
#         # Quantum excitation
#         if use_excitation
#             quantum_excitation!(E0, radius, σ_E0, buffers, particles)
#         end
        
#         # Synchrotron radiation damping
#         if SR_damping
#             synchrotron_radiation!(E0, radius, particles)
#         end
        
#         # Apply wakefield effects
#         if use_wakefield
#             particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
#             curr = particle_factor / E0 / (2*π*radius) * σ_z / (η0 * σ_E0^2)
            
#             apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, curr, σ_z, bin_edges)
            
#             if update_E0
#                 # Update reference energy based on collective effects
#                 mean_ΔE_diff = compute_mean_diff(particles.coordinates.ΔE, ΔE_before)
#                 E0 = E0 + mean_ΔE_diff

#                 # Zero the mean energy deviation - without allocation
#                 mean_ΔE = compute_mean(particles.coordinates.ΔE)
#                 subtract_mean_inplace!(particles.coordinates.ΔE, mean_ΔE)
#             end
#         end
        
#         # Update reference energy if needed
#         if update_E0
#             γ0 = E0 / mass
#             β0 = 1 - 1 / γ0 / γ0
#             η0 = α_c -  1 / γ0 / γ0
            
#             # Adjust for radiation losses
#             if SR_damping
#                 E0 = E0 - 4 * 8.85e-5 * (E0/1e9)^3 / radius * E0 / 4
#                 γ0 = E0 / mass
#                 β0 = 1 - 1 / γ0 / γ0
#             end
#         end
        
#         # Update phase advance
#         if update_η
#             @turbo for i in 1:n_particles
#                 # Calculate slip factor for each particle
#                 Δγ_i = particles.coordinates.ΔE[i] / mass
#                 η_i = α_c - 1/(γ0 + Δγ_i)^2
                
#                 # Inline phase advance calculations
#                 coeff_i = 2π * harmonic * η_i / (β0 * β0 * E0)
#                 ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs) # Inlined z_to_ϕ
#                 ϕ_i += coeff_i * particles.coordinates.ΔE[i]
#                 particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor # Inlined ϕ_to_z
#             end
#         else
#             # Using constant slip factor
#             apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
#         end
        
#         # Update RF factor with new beta
#         rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        
#         # Update progress
#         if show_progress
#             next!(p)
#         end
#     end
    
#     # Final spreads
#     σ_E = compute_std(particles.coordinates.ΔE)
#     σ_z = compute_std(particles.coordinates.z)
    
#     return σ_E, σ_z, E0
# end

# """
#     apply_phase_advance!(
#         particles::StructArray{Particle{T}},
#         η0,
#         harmonic,
#         β0,
#         E0,
#         rf_factor,
#         ϕs
#     ) where T<:Float64 -> Nothing

# Apply phase advancement to all particles.
# Memory-efficient implementation for StochasticTriple.
# """
# function apply_phase_advance!(
#     particles::StructArray{Particle{T}},
#     η0,
#     harmonic,
#     β0,
#     E0,
#     rf_factor,
#     ϕs
# ) where T<:Float64
    
#     # # Check type once, outside the loop
#     # is_stochastic = any(typeof(param) <: StochasticTriple for param in 
#     #                     [η0, harmonic, β0, E0, rf_factor, ϕs])
    
#     # if is_stochastic
#     #     # Calculate coefficient once - single propagate call
#     #     coeff_fn = (_η0, _harmonic, _β0, _E0) -> begin
#     #         return 2π * _harmonic * _η0 / (_β0 * _β0 * _E0)
#     #     end
        
#     #     coeff = StochasticAD.propagate(coeff_fn, η0, harmonic, β0, E0)
        
#     #     # Helper for phase space conversion
#     #     phase_fn = (_rf_factor, _ϕs, z_i, coeff_val, ΔE_i) -> begin
#     #         ϕ_i = z_to_ϕ(z_i, _rf_factor, _ϕs)
#     #         ϕ_i += coeff_val * ΔE_i
#     #         return ϕ_to_z(ϕ_i, _rf_factor, _ϕs)
#     #     end
        
#     #     # Process all particles
#     #     for i in 1:length(particles)
#     #         # Fix: Use particles.coordinates.z directly instead of z_values
#     #         particles.coordinates.z[i] = StochasticAD.propagate(
#     #             (_rf, _ϕs) -> phase_fn(_rf, _ϕs, particles.coordinates.z[i], coeff, particles.coordinates.ΔE[i]),
#     #             rf_factor, ϕs
#     #         )
#     #     end
#     # else
#     #     # Standard implementation - vectorized calculation
#     #     coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
#     #     # Process all particles
#     #     for i in 1:length(particles)
#     #         ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
#     #         ϕ_i += coeff * particles.coordinates.ΔE[i]
#     #         particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
#     #     end
#     # end

#     coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
#     # Process all particles - inlined z_to_ϕ and ϕ_to_z
#     @turbo for i in 1:length(particles)
#         ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs)
#         ϕ_i += coeff * particles.coordinates.ΔE[i]
#         particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor
#     end

# end

# function safe_update_energy!(particles::StructArray{Particle{T}}, mean_value) where T<:Float64
#     # If mean_value is a StochasticTriple, we need special handling
#     if typeof(mean_value) <: StochasticTriple
#         for i in 1:length(particles)
#             # Use propagate to handle the subtraction properly
#             particles.coordinates.ΔE[i] = StochasticAD.propagate(
#                 (e, m) -> e - m,
#                 particles.coordinates.ΔE[i],
#                 mean_value
#             )
#         end
#     else
#         # If it's a regular Float64, just do the subtraction directly
#         for i in 1:length(particles)
#             particles.coordinates.ΔE[i] -= mean_value
#         end
#     end
#     return nothing
# end

# File: evolution.jl

"""
evolution.jl - Particle evolution and beam dynamics (Particle Distribution Strategy)

Implements the core longitudinal evolution algorithm and particle generation.
Each rank manages a subset of particles. Standard physics steps are local.
Wakefield involves MPI communication (Allreduce histogram, Bcast potential).
Includes MPI logic for global E0 updates.
"""

using StructArrays
using StaticArrays
using Statistics
using Random
using Distributions
using ProgressMeter
using LoopVectorization
using MPI # Add MPI
using LinearAlgebra # For I in generate_particles

# --- generate_particles (Definition needed, even if called locally per rank) ---
"""
    generate_particles(...)

Generate initial particle distribution. Called locally per rank in the particle distribution strategy.
Handles edge case of num_particles < 2.
"""
function generate_particles(
    μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int, # num_particles is now n_local
    energy::T, mass::T, ϕs::T, freq_rf::T) where T<:Float64

    # Ensure num_particles is non-negative
    if num_particles <= 0
         # Return an empty StructArray of the correct type and zero spreads
         return StructArray{Particle{Float64}}(undef, 0), T(0), T(0), energy
    end

    z_vals = Vector{T}(undef, num_particles)
    ΔE_vals = Vector{T}(undef, num_particles)
    actual_σ_E::T = 0.0
    actual_σ_z::T = 0.0

    # Need at least 2 samples for covariance calculation
    if num_particles < 2
        z_vals[1] = rand(Normal(μ_z, σ_z))
        ΔE_vals[1] = rand(Normal(μ_E, σ_E))
        actual_σ_E = σ_E # Cannot calculate std dev, return input
        actual_σ_z = σ_z
    else
         initial_sample_size::Int = min(num_particles, 10_000)
         z_samples = rand(Normal(μ_z, σ_z), initial_sample_size)
         E_samples = rand(Normal(μ_E, σ_E), initial_sample_size)

         local Σ
         try # Calculate covariance
              Σ = Symmetric([cov(z_samples, z_samples) cov(z_samples, E_samples);
                             cov(z_samples, E_samples) cov(E_samples, E_samples)])
         catch e
              @warn "Covariance calculation failed: $e. Using diagonal matrix."
              Σ = Symmetric(diagm([max(σ_z^2, T(1e-18)), max(σ_E^2, T(1e-18))]))
         end

         # Ensure positive semi-definite
         min_eig = 0.0
         try min_eig = eigmin(Σ) catch; min_eig = -Inf; end
         if min_eig < 1e-12; Σ = Σ + (1e-12 - min_eig) * I; end

         # Create distribution
         μ = SVector{2,T}(μ_z, μ_E)
         local dist_total
         try dist_total = MvNormal(μ, Σ) catch e; @error "MvNormal creation failed: $e"; throw(e); end

         # Generate samples
         samples = rand(dist_total, num_particles)
         z_vals .= samples[1, :]
         ΔE_vals .= samples[2, :]

         # Calculate actual spread from generated samples
         actual_σ_E = std(ΔE_vals)
         actual_σ_z = std(z_vals)
    end

    particles = StructArray{Particle{Float64}}((
        coordinates=StructArray(Coordinate.(z_vals, ΔE_vals)),
    ))
    return particles, actual_σ_E, actual_σ_z, energy
end


# --- Helper Function for Global Standard Deviation ---
""" Calculate global standard deviation via MPI Allreduce """
function compute_global_std(local_data::AbstractVector{T}, comm::MPI.Comm) where T<:Float64
    n_local = length(local_data)
    sum_local = zero(T); sum_sq_local = zero(T)
    if n_local > 0
        @simd for x in local_data; sum_local += x; sum_sq_local += x * x; end
    end
    sums = MPI.Allreduce([sum_local, sum_sq_local, T(n_local)], MPI.SUM, comm)
    sum_global = sums[1]; sum_sq_global = sums[2]; n_global = Int(round(sums[3]))
    if n_global <= 1; return zero(T); end
    mean_global = sum_global / n_global
    variance_global = max(zero(T), (sum_sq_global / n_global) - (mean_global * mean_global))
    return sqrt(variance_global * n_global / (n_global - 1))
end

# --- Helper Function for Global Mean ---
""" Calculate global mean via MPI Allreduce """
function compute_global_mean(local_data::AbstractVector{T}, comm::MPI.Comm) where T<:Float64
     n_local = length(local_data)
     sum_local = zero(T)
     if n_local > 0
         @simd for x in local_data; sum_local += x; end
     end
     sums = MPI.Allreduce([sum_local, T(n_local)], MPI.SUM, comm)
     sum_global = sums[1]; n_global = Int(round(sums[2]))
     return n_global > 0 ? sum_global / n_global : zero(T)
end


"""
    longitudinal_evolve!(...) - Particle Distribution Version with Global E0 Update

Evolve particles. Each rank evolves its local subset.
Wakefield calculation requires MPI communication. E0 updated globally.
"""
function longitudinal_evolve!(
    particles::StructArray{Particle{T}}, # Now holds N_local particles
    params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
    buffers::SimulationBuffers{T},      # Now sized for N_local / nbins
    ; show_progress::Bool = true
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # --- Parameters (all ranks have them) ---
    # E0 is now treated as a mutable global state, updated each turn
    local E0::TE = params.E0 # Use local mutable variable, initialized from params
    mass = params.mass
    voltage = params.voltage
    harmonic = params.harmonic
    radius = params.radius;
    freq_rf = params.freq_rf
    pipe_radius = params.pipe_radius
    α_c = params.α_c
    ϕs = params.ϕs;
    n_turns = params.n_turns
    use_wakefield = params.use_wakefield
    update_η = params.update_η
    update_E0_global = params.update_E0 # Use the flag from parameters
    SR_damping = params.SR_damping; use_excitation = params.use_excitation

    # --- Local Particle Count ---
    n_local = length(particles)

    # --- Initial GLOBAL Spreads (for scales) ---
    σ_E0_global = compute_global_std(particles.coordinates.ΔE, comm)
    σ_z0_global = compute_global_std(particles.coordinates.z, comm)

    # --- Pre-calculate fixed bin_edges and wake parameters (All Ranks if wakefield used) ---
    local bin_edges::AbstractRange{T} = 1.0:2.0
    local wake_factor_val::T = 0.0; local wake_sqrt_val::T = 0.0; local cτ::T = 0.0; local inv_cτ::T = 0.0
    if use_wakefield
        nbins::Int = length(buffers.λ); if nbins <= 0; error("Buffer nbins invalid."); end
        bin_edges = range(-7.5*σ_z0_global, 7.5*σ_z0_global, length=nbins+1)
        kp=T(3e1); Z0=T(120π); cτ=T(4e-3); if cτ <= 0 error("cτ invalid."); end
        inv_cτ = 1.0 / cτ; if pipe_radius <= 0 error("pipe_radius invalid."); end
        wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
        wake_sqrt_val = sqrt(max(T(0), 2 * kp / pipe_radius))
    end

    # Progress Meter (Only Rank 0)
    local p::Progress
    if rank == 0 && show_progress; p = Progress(n_turns, desc="Simulating Turns (Dist): "); end

    # Buffer for initial ΔE (needed for E0 update) - allocate on all ranks
    ΔE_initial_turn = Vector{T}(undef, n_local)

    # --- Main Evolution Loop (Executed by All Ranks on Local Data) ---
    for turn in 1:n_turns

        # --- Calculate factors based on CURRENT E0 (All Ranks) ---
        # E0 is updated at the *end* of the loop, so use current value here
        local γ0 = E0 / mass
        local β0 = sqrt(max(0.0, 1.0 - 1.0 / (γ0 * γ0)))
        local η0 = α_c - 1.0 / (γ0 * γ0) : α_c
        local sin_ϕs = sin(ϕs)
        local rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)

        # --- Store Initial ΔE State ---
        if update_E0_global && n_local > 0
            copyto!(ΔE_initial_turn, particles.coordinates.ΔE)
        end

        # --- RF Kick (Local) ---
        if n_local > 0
            f_kick!(voltage, sin_ϕs, rf_factor, ϕs, particles)
        end

        # --- Quantum Excitation (Local) ---
        if use_excitation && n_local > 0
            quantum_excitation!(E0, radius, σ_E0_global, buffers, particles)
        end

        # --- SR Damping (Local) ---
        if SR_damping && n_local > 0
            synchrotron_radiation!(E0, radius, particles)
        end

        # --- Phase Advance (Local) ---
         if n_local > 0
             if update_η
                 @turbo for i in 1:n_local
                     Δγ_i = particles.coordinates.ΔE[i] / mass
                     denom_sq = (γ0 + Δγ_i)^2
                     η_i = α_c - 1.0 / denom_sq 
                     coeff_i = (2π * harmonic * η_i / (β0 * β0 * E0)) 
                     ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs) 
                     ϕ_i += coeff_i * particles.coordinates.ΔE[i]
                     particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor 
                 end
             else
                 apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
             end
         end

        # --- Wakefield Calculation (Requires Communication) ---
        if use_wakefield
            current_σ_z_global = compute_global_std(particles.coordinates.z, comm)
            n_particles_global = MPI.Allreduce(n_local, MPI.SUM, comm) # Get accurate global N

            local current_wake_current::T = 0.0
            local current_γ0_wake = E0 / mass  # Use E0 before potential E0 update this turn
            local current_η0_wake =  α_c - 1.0 / (current_γ0_wake * current_γ0_wake) 
            if E0 > 0 && abs(current_η0_wake) > 1e-18 && σ_E0_global > 1e-18 && n_particles_global > 0
                log10_N = log10(n_particles_global)
                floor_log10_N = floor(Int, log10_N)
                power_val = max(-10, min(10, floor_log10_N))
                denominator_pf = 10.0^power_val
                if denominator_pf == 0 
                    denominator_pf = 1.0 
                end
                particle_factor = (1e11 / denominator_pf ) * n_particles_global
                current_wake_current = particle_factor / E0 / (2*π*radius) * current_σ_z_global / (current_η0_wake * σ_E0_global^2)
            end

            # All ranks call wakefield function
            apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, current_wake_current, current_σ_z_global, bin_edges)
            # Modifies local ΔE based on broadcasted potential
        end

        # --- Global E0 Update (Requires Communication) ---
        if update_E0_global
            local sum_dE_local = zero(T)
            if n_local > 0
                # Calculate sum of ΔE changes on this rank
                @simd for i in 1:n_local
                    sum_dE_local += (particles.coordinates.ΔE[i] - ΔE_initial_turn[i])
                end
            end

            # Reduce sum of changes and local counts
            reductions = MPI.Allreduce([sum_dE_local, T(n_local)], MPI.SUM, comm)
            sum_dE_global = reductions[1]
            n_global = Int(round(reductions[2]))

            # Calculate global mean energy change
            mean_dE_global = n_global > 0 ? sum_dE_global / n_global : zero(T)

            # All ranks update E0 identically
            E0 = E0 + mean_dE_global

            # Re-center local particle energies relative to the NEW E0
            # by subtracting the average change we just added to E0
            if n_local > 0
                particles.coordinates.ΔE .-= mean_dE_global
            end

            # Note: E0 derived factors (gamma0, beta0, etc.) will be recalculated
            # at the START of the *next* turn using this updated E0.
        end


        # --- Update Progress Meter (Rank 0) ---
        if rank == 0 && show_progress; next!(p); end

    end # End turn loop

    # --- Final GLOBAL Spreads (for returning values) ---
    σ_E_final = compute_global_std(particles.coordinates.ΔE, comm)
    σ_z_final = compute_global_std(particles.coordinates.z, comm)

    # Return global spreads and final E0 (consistent across all ranks)
    return σ_E_final, σ_z_final, E0
end


"""
    apply_phase_advance!(...) - Local version
"""
function apply_phase_advance!(
    particles::StructArray{Particle{T}}, # Local particles
    η0, harmonic, β0, E0, rf_factor, ϕs
) where T<:Float64
    n_local = length(particles)
    if n_local == 0 return nothing end
    local coeff = (2π * harmonic * η0 / (β0 * β0 * E0))
    @turbo for i in 1:n_local
        ϕ_i =  -(particles.coordinates.z[i] * rf_factor - ϕs)
        ϕ_i += coeff * particles.coordinates.ΔE[i]
        particles.coordinates.z[i] =  (-ϕ_i + ϕs) / rf_factor 
    end
    return nothing
end

"""
    safe_update_energy!(...) - Local version
"""
function safe_update_energy!(particles::StructArray{Particle{T}}, value_to_subtract) where T<:Float64
     n_local = length(particles)
     if n_local == 0 return nothing end
     @turbo for i in 1:n_local; particles.coordinates.ΔE[i] -= value_to_subtract; end
     return nothing
end

# Assume rf_kick!, quantum_excitation!, synchrotron_radiation! are defined elsewhere
# and operate locally on the passed `particles` StructArray.