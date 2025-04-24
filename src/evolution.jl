"""
evolution.jl - Particle evolution and beam dynamics

This file implements the core longitudinal evolution algorithm and particle generation.
It handles the multi-turn tracking of particles through the accelerator,
including RF cavity effects, synchrotron radiation, and collective effects.
"""

using StructArrays
using StaticArrays
using Statistics
using Random
using Distributions
using ProgressMeter
using LoopVectorization

"""
    generate_particles(
        μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
        energy::T, mass::T, ϕs::T, freq_rf::T
    ) where T<:Float64 -> Tuple{StructArray{Particle{T}}, T, T, T}

Generate initial particle distribution.
"""
function generate_particles(
    μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
    energy::T, mass::T, ϕs::T, freq_rf::T) where T<:Float64

    # Initial sampling for covariance estimation
    initial_sample_size::Int = min(10_000, num_particles)
    z_samples = rand(Normal(μ_z, σ_z), initial_sample_size)
    E_samples = rand(Normal(μ_E, σ_E), initial_sample_size)

    # Compute covariance matrix
    Σ = Symmetric([cov(z_samples, z_samples) cov(z_samples, E_samples);
                   cov(z_samples, E_samples) cov(E_samples, E_samples)])

    # Create multivariate normal distribution
    μ = SVector{2,T}(μ_z, μ_E)
    dist_total = MvNormal(μ, Σ)

    # Relativistic factors
    γ::T = energy / mass
    β::T = sqrt(1 - 1/γ^2)
    rf_factor::T = freq_rf * 2π / (β * SPEED_LIGHT)

    # Generate correlated random samples
    samples = rand(dist_total, num_particles)  # 2 × num_particles matrix
    z_vals = samples[1, :]
    ΔE_vals = samples[2, :]


    # Create the StructArray of Particles

    # particles = StructArray{Particle{Float64}}(StructArray(Coordinate.(z_vals, ΔE_vals)))


    particles = StructArray{Particle{Float64}}((
    StructArray(Coordinate.(z_vals, ΔE_vals)),  # coordinates
    # StructArray(Coordinate.(zeros(num_particles), zeros(num_particles)))  # uncertainty
    ))


    return particles, σ_E, σ_z, energy
end


"""
    longitudinal_evolve!(
        particles::StructArray{Particle{T}},
        params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
        buffers::SimulationBuffers{T}
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF} -> Tuple{T, T, TE}

Evolve particles through the accelerator for multiple turns.
Type-stable implementation supporting StochasticTriple parameters.

Returns: (σ_E, σ_z, E0) - Final energy spread, bunch length, and reference energy
"""
# function longitudinal_evolve!(
#     particles::StructArray{Particle{T}},
#     params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
#     buffers::SimulationBuffers{T}
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
#     # Use StochasticAD.propagate for all calculations that might involve StochasticTriple
#     # γ0 = StochasticAD.propagate((energy, m) -> energy / m, E0, mass)
#     # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
#     # η0 = StochasticAD.propagate((alpha, gamma) -> alpha - 1/(gamma^2), α_c, γ0)
#     γ0 = E0 / mass
#     β0 = 1 - 1 / γ0 / γ0
#     η0 = α_c -  1 / γ0 / γ0
#     sin_ϕs = sin(ϕs)
#     # rf_factor = StochasticAD.propagate((freq, beta) -> freq * 2π / (beta * SPEED_LIGHT), freq_rf, β0)
#     rf_factor = calc_rf_factor(freq_rf, β0)
    
#     # Get initial spreads
#     n_particles::Int = length(particles)
#     σ_E0::T = std(particles.coordinates.ΔE)
#     σ_z0::T = std(particles.coordinates.z)
    
#     if use_wakefield
#         nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
#         bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
#         kp = convert(T, 3e1)
#         Z0 = convert(T, 120π)
#         cτ = convert(T, 4e-3)
        
#         # For wake calculations, safely convert values using propagate
#         # wake_factor_val = StochasticAD.propagate(r -> Z0 * SPEED_LIGHT / (π * r), pipe_radius)
#         # wake_sqrt_val = StochasticAD.propagate(r -> sqrt(2 * kp / r), pipe_radius)

#         wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
#         wake_sqrt_val = sqrt(2 * kp / pipe_radius)
#     end
    
#     # Setup progress meter
#     p = Progress(n_turns, desc="Simulating Turns: ")
    
#     # Main evolution loop
#     for turn in 1:n_turns
#         # Calculate current spreads
#         σ_E::T = std(particles.coordinates.ΔE)
#         σ_z::T = std(particles.coordinates.z)
        
#         # Store previous energy values for update_E0 if needed
#         if update_E0
#             ΔE_before = copy(particles.coordinates.ΔE)
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
#             # Use propagate for all calculations that might involve StochasticTriple
#             # curr = StochasticAD.propagate(
#             #     (eta, energy, rad) -> begin
#             #         # Avoid explicit Float64 conversions
#             #         particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
#             #         return particle_factor / energy / (2*π*rad) * σ_z / (eta * σ_E0^2)
#             #     end,
#             #     η0, E0, radius
#             # )
#             particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
#             curr = particle_factor / E0 / (2*π*radius) * σ_z / (η0 * σ_E0^2)
            
#             apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, curr, σ_z, bin_edges)
            
#             if update_E0
#                 # Update reference energy based on collective effects
#                 mean_ΔE_diff = mean(particles.coordinates.ΔE .- ΔE_before)
#                 # E0 = StochasticAD.propagate((energy, diff) -> energy + diff, E0, mean_ΔE_diff)
#                 E0 = E0 + mean_ΔE_diff

#                 # Zero the mean energy deviation - FIXED VERSION
#                 mean_ΔE = mean(particles.coordinates.ΔE)
#                 # Use the safe update function instead of direct assignment
#                 safe_update_energy!(particles, mean_ΔE)
#             end
#         end
        
#         # Update reference energy if needed
#         if update_E0
#             # Use propagate for all operations that involve potential StochasticTriples
#             # E0 = StochasticAD.propagate((energy, v, s) -> energy + v * s, E0, voltage, sin_ϕs)
#             # γ0 = StochasticAD.propagate((energy, m) -> energy/m, E0, mass)
#             # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)

#             γ0 = E0 / mass
#             β0 = 1 - 1 / γ0 / γ0
#             η0 = α_c -  1 / γ0 / γ0
            
#             # Adjust for radiation losses
#             if SR_damping
#                 # E0 = StochasticAD.propagate(
#                 #     (energy, rad) -> begin
#                 #         # Calculate radiation coefficient without explicit type conversion
#                 #         radiation_coeff = 4 * 8.85e-5 * (energy/1e9)^3 / rad
#                 #         # Apply energy loss
#                 #         return energy - radiation_coeff * energy / 4
#                 #     end,
#                 #     E0, radius
#                 # )

#                 E0 = E0 - 4 * 8.85e-5 * (E0/1e9)^3 / radius * E0 / 4
#                 # γ0 = StochasticAD.propagate((energy, m) -> energy/m, E0, mass)
#                 # β0 = StochasticAD.propagate(γ -> sqrt(1 - 1/γ^2), γ0)
#                 γ0 = E0 / mass
#                 β0 = 1 - 1 / γ0 / γ0
#             end
#         end
        
#         # Update phase advance
#         if update_η
#             for i in 1:n_particles
#                 # Calculate slip factor for each particle using propagate
#                 Δγ_i = particles.coordinates.ΔE[i] / mass
#                 # η_i = StochasticAD.propagate(
#                 #     (alpha, gamma, delta_gamma) -> alpha - 1/(gamma + delta_gamma)^2,
#                 #     α_c, γ0, Δγ_i
#                 # )
#                 η_i = α_c - 1/(γ0 + Δγ_i)^2
#                 # Use helper function for phase advance with proper StochasticTriple handling
#                 # particles.coordinates.z[i] = StochasticAD.propagate(
#                 #     (eta_i, h, beta, energy, rf, phi_s) -> begin
#                 #         coeff_i = 2π * h * eta_i / (beta * beta * energy)
#                 #         ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf, phi_s)
#                 #         ϕ_i += coeff_i * particles.coordinates.ΔE[i]
#                 #         return ϕ_to_z(ϕ_i, rf, phi_s)
#                 #     end,
#                 #     η_i, harmonic, β0, E0, rf_factor, ϕs
#                 # )

#                 coeff_i = 2π * harmonic * η_i / (β0 * β0 * E0)
#                 ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
#                 ϕ_i += coeff_i * particles.coordinates.ΔE[i]
#                 particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)

#             end
#         else
#             # Using constant slip factor - use the helper function
#             apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
#         end
        
#         # Update RF factor with new beta
#         # rf_factor = StochasticAD.propagate(
#         #     (freq, beta) -> freq * 2π / (beta * SPEED_LIGHT),
#         #     freq_rf, β0
#         # )
#         rf_factor = calc_rf_factor(freq_rf, β0)
#         # Update progress
#         next!(p)
#     end
    
#     # Final spreads
#     σ_E = std(particles.coordinates.ΔE)
#     σ_z = std(particles.coordinates.z)
    
#     # Return the actual E0 type without conversion
#     return σ_E, σ_z, E0
# end

function longitudinal_evolve!(
    particles::StructArray{Particle{T}},
    params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
    buffers::SimulationBuffers{T}
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}
    
    # Extract parameters
    E0 = params.E0
    mass = params.mass
    voltage = params.voltage
    harmonic = params.harmonic
    radius = params.radius
    freq_rf = params.freq_rf
    pipe_radius = params.pipe_radius
    α_c = params.α_c
    ϕs = params.ϕs
    n_turns = params.n_turns
    use_wakefield = params.use_wakefield
    update_η = params.update_η
    update_E0 = params.update_E0
    SR_damping = params.SR_damping
    use_excitation = params.use_excitation
    
    # Pre-compute physical constants - handle StochasticTriple values
    γ0 = E0 / mass
    β0 = 1 - 1 / γ0 / γ0
    η0 = α_c -  1 / γ0 / γ0
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Get initial spreads
    n_particles::Int = length(particles)
    
    # Pre-compute standard deviations once before main loop
    σ_E0::T = compute_std(particles.coordinates.ΔE)
    σ_z0::T = compute_std(particles.coordinates.z)
    
    # Initialize wake parameters
    if use_wakefield
        nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
        bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
        kp = convert(T, 3e1)
        Z0 = convert(T, 120π)
        cτ = convert(T, 4e-3)
        
        wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
        wake_sqrt_val = sqrt(2 * kp / pipe_radius)
    end
    
    # Setup progress meter
    p = Progress(n_turns, desc="Simulating Turns: ")
    
    # Pre-allocate energy buffer for updates
    ΔE_before = Vector{T}(undef, n_particles)
    
    # Main evolution loop
    for turn in 1:n_turns
        # Calculate current spreads - inline calculation to avoid temp arrays
        σ_E::T = compute_std(particles.coordinates.ΔE) 
        σ_z::T = compute_std(particles.coordinates.z)
        
        # Store previous energy values for update_E0 if needed
        if update_E0
            copyto!(ΔE_before, particles.coordinates.ΔE)
        end
        
        # RF voltage kick
        rf_kick!(voltage, sin_ϕs, rf_factor, ϕs, particles)
        
        # Quantum excitation
        if use_excitation
            quantum_excitation!(E0, radius, σ_E0, buffers, particles)
        end
        
        # Synchrotron radiation damping
        if SR_damping
            synchrotron_radiation!(E0, radius, particles)
        end
        
        # Apply wakefield effects
        if use_wakefield
            particle_factor = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
            curr = particle_factor / E0 / (2*π*radius) * σ_z / (η0 * σ_E0^2)
            
            apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, curr, σ_z, bin_edges)
            
            if update_E0
                # Update reference energy based on collective effects
                mean_ΔE_diff = compute_mean_diff(particles.coordinates.ΔE, ΔE_before)
                E0 = E0 + mean_ΔE_diff

                # Zero the mean energy deviation - without allocation
                mean_ΔE = compute_mean(particles.coordinates.ΔE)
                subtract_mean_inplace!(particles.coordinates.ΔE, mean_ΔE)
            end
        end
        
        # Update reference energy if needed
        if update_E0
            γ0 = E0 / mass
            β0 = 1 - 1 / γ0 / γ0
            η0 = α_c -  1 / γ0 / γ0
            
            # Adjust for radiation losses
            if SR_damping
                E0 = E0 - 4 * 8.85e-5 * (E0/1e9)^3 / radius * E0 / 4
                γ0 = E0 / mass
                β0 = 1 - 1 / γ0 / γ0
            end
        end
        
        # Update phase advance
        if update_η
            @turbo for i in 1:n_particles
                # Calculate slip factor for each particle
                Δγ_i = particles.coordinates.ΔE[i] / mass
                η_i = α_c - 1/(γ0 + Δγ_i)^2
                
                # Inline phase advance calculations
                coeff_i = 2π * harmonic * η_i / (β0 * β0 * E0)
                ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs) # Inlined z_to_ϕ
                ϕ_i += coeff_i * particles.coordinates.ΔE[i]
                particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor # Inlined ϕ_to_z
            end
        else
            # Using constant slip factor
            apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
        end
        
        # Update RF factor with new beta
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        
        # Update progress
        next!(p)
    end
    
    # Final spreads
    σ_E = compute_std(particles.coordinates.ΔE)
    σ_z = compute_std(particles.coordinates.z)
    
    return σ_E, σ_z, E0
end

"""
    apply_phase_advance!(
        particles::StructArray{Particle{T}},
        η0,
        harmonic,
        β0,
        E0,
        rf_factor,
        ϕs
    ) where T<:Float64 -> Nothing

Apply phase advancement to all particles.
Memory-efficient implementation for StochasticTriple.
"""
function apply_phase_advance!(
    particles::StructArray{Particle{T}},
    η0,
    harmonic,
    β0,
    E0,
    rf_factor,
    ϕs
) where T<:Float64
    
    # # Check type once, outside the loop
    # is_stochastic = any(typeof(param) <: StochasticTriple for param in 
    #                     [η0, harmonic, β0, E0, rf_factor, ϕs])
    
    # if is_stochastic
    #     # Calculate coefficient once - single propagate call
    #     coeff_fn = (_η0, _harmonic, _β0, _E0) -> begin
    #         return 2π * _harmonic * _η0 / (_β0 * _β0 * _E0)
    #     end
        
    #     coeff = StochasticAD.propagate(coeff_fn, η0, harmonic, β0, E0)
        
    #     # Helper for phase space conversion
    #     phase_fn = (_rf_factor, _ϕs, z_i, coeff_val, ΔE_i) -> begin
    #         ϕ_i = z_to_ϕ(z_i, _rf_factor, _ϕs)
    #         ϕ_i += coeff_val * ΔE_i
    #         return ϕ_to_z(ϕ_i, _rf_factor, _ϕs)
    #     end
        
    #     # Process all particles
    #     for i in 1:length(particles)
    #         # Fix: Use particles.coordinates.z directly instead of z_values
    #         particles.coordinates.z[i] = StochasticAD.propagate(
    #             (_rf, _ϕs) -> phase_fn(_rf, _ϕs, particles.coordinates.z[i], coeff, particles.coordinates.ΔE[i]),
    #             rf_factor, ϕs
    #         )
    #     end
    # else
    #     # Standard implementation - vectorized calculation
    #     coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
    #     # Process all particles
    #     for i in 1:length(particles)
    #         ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
    #         ϕ_i += coeff * particles.coordinates.ΔE[i]
    #         particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
    #     end
    # end

    coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
    # Process all particles - inlined z_to_ϕ and ϕ_to_z
    @turbo for i in 1:length(particles)
        ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs)
        ϕ_i += coeff * particles.coordinates.ΔE[i]
        particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor
    end

end

function safe_update_energy!(particles::StructArray{Particle{T}}, mean_value) where T<:Float64
    # If mean_value is a StochasticTriple, we need special handling
    if typeof(mean_value) <: StochasticTriple
        for i in 1:length(particles)
            # Use propagate to handle the subtraction properly
            particles.coordinates.ΔE[i] = StochasticAD.propagate(
                (e, m) -> e - m,
                particles.coordinates.ΔE[i],
                mean_value
            )
        end
    else
        # If it's a regular Float64, just do the subtraction directly
        for i in 1:length(particles)
            particles.coordinates.ΔE[i] -= mean_value
        end
    end
    return nothing
end