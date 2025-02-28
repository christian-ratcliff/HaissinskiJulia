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
Random.seed!(1234)
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
    StructArray(Coordinate.(zeros(num_particles), zeros(num_particles)))  # uncertainty
    ))


    return particles, σ_E, σ_z, energy
end


"""
    longitudinal_evolve!(
        particles::StructArray{Particle{T}},
        params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
        buffers::SimulationBuffers{T}
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF} -> Tuple{T, T, T}

Evolve particles through the accelerator for multiple turns.
Type-stable implementation supporting StochasticTriple parameters.

Returns: (σ_E, σ_z, E0) - Final energy spread, bunch length, and reference energy
"""
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
    
    # Pre-compute physical constants
    γ0 = E0 / mass
    β0 = sqrt(1 - 1/γ0^2)
    η0 = α_c - 1/(γ0^2)
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Get initial spreads
    n_particles::Int = length(particles)
    σ_E0::T = std(particles.coordinates.ΔE)
    σ_z0::T = std(particles.coordinates.z)
    
    if use_wakefield
        nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
        bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
        kp = convert(T, 3e1)
        Z0 = convert(T, 120π)
        cτ = convert(T, 4e-3)
        
        # Convert StochasticTriple types to Float64 only where needed for specific calculations
        wake_factor_val = Z0 * SPEED_LIGHT / (π * convert(T, pipe_radius))
        wake_sqrt_val = sqrt(2 * kp / convert(T, pipe_radius))
    end
    
    # Setup progress meter
    p = Progress(n_turns, desc="Simulating Turns: ")
    
    # Main evolution loop
    for turn in 1:n_turns
        # Calculate current spreads
        σ_E::T = std(particles.coordinates.ΔE)
        σ_z::T = std(particles.coordinates.z)
        
        # Store previous energy values for update_E0 if needed
        if update_E0
            ΔE_before = copy(particles.coordinates.ΔE)
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
            # For complex calculations with mixed types, convert to Float64 as needed
            η0_val = convert(T, η0)
            radius_val = convert(T, radius)
            E0_val = convert(T, E0)
            
            curr = convert(T, (1e11/(10.0^floor(Int, log10(n_particles))))) * 
                  convert(T, n_particles) / E0_val / (2*π*radius_val) * 
                  σ_z / (η0_val * σ_E0^2)
                  
            apply_wakefield_inplace!(particles, buffers, wake_factor_val, wake_sqrt_val, cτ, curr, σ_z, bin_edges)
            
            if update_E0
                # Update reference energy based on collective effects
                E0 += mean(particles.coordinates.ΔE .- ΔE_before)
                
                # Zero the mean energy deviation
                mean_ΔE = mean(particles.coordinates.ΔE)
                for i in 1:n_particles
                    particles.coordinates.ΔE[i] -= mean_ΔE
                end
            end
        end
        
        # Update reference energy if needed
        if update_E0
            E0 += voltage * sin_ϕs
            γ0 = E0/mass 
            β0 = sqrt(1 - 1/γ0^2)
            
            # Adjust for radiation losses
            if SR_damping
                E0_val = convert(T, E0)
                radius_val = convert(T, radius)
                ∂U_∂E = convert(T, 4 * 8.85e-5) * (E0_val/1e9)^3 / radius_val
                E0 -= ∂U_∂E * E0 / 4
                γ0 = E0/mass 
                β0 = sqrt(1 - 1/γ0^2)
            end
        end
        
        # Update phase advance
        if update_η
            for i in 1:n_particles
                # Calculate slip factor for each particle
                Δγ_i = particles.coordinates.ΔE[i] / mass
                η_i = α_c - 1/(γ0 + Δγ_i)^2
                
                # Use helper function for phase advance with proper StochasticTriple handling
                particles.coordinates.z[i] = StochasticAD.propagate(
                    (_η_i, _harmonic, _β0, _E0, _rf_factor, _ϕs) -> begin
                        coeff_i = 2π * _harmonic * _η_i / (_β0 * _β0 * _E0)
                        ϕ_i = z_to_ϕ(particles.coordinates.z[i], _rf_factor, _ϕs)
                        ϕ_i += coeff_i * particles.coordinates.ΔE[i]
                        return ϕ_to_z(ϕ_i, _rf_factor, _ϕs)
                    end,
                    η_i, harmonic, β0, E0, rf_factor, ϕs
                )
            end
        else
            # Using constant slip factor - use the helper function
            apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
        end
        
        # Update RF factor with new beta
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        
        # Update progress
        next!(p)
    end
    
    # Final spreads
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    
    # Convert final E0 to Float64 for return
    E0_final = convert(T, E0)
    
    return σ_E, σ_z, E0_final
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
    
    # Check type once, outside the loop
    is_stochastic = any(typeof(param) <: StochasticTriple for param in 
                        [η0, harmonic, β0, E0, rf_factor, ϕs])
    
    if is_stochastic
        # Calculate coefficient once - single propagate call
        coeff_fn = (_η0, _harmonic, _β0, _E0) -> begin
            return 2π * _harmonic * _η0 / (_β0 * _β0 * _E0)
        end
        
        coeff = StochasticAD.propagate(coeff_fn, η0, harmonic, β0, E0)
        
        # Extract raw arrays for faster processing
        # z_values = particles.coordinates.z
        # ΔE_values = particles.coordinates.ΔE
        
        # Helper for phase space conversion
        phase_fn = (_rf_factor, _ϕs, z_i, coeff_val, ΔE_i) -> begin
            ϕ_i = z_to_ϕ(z_i, _rf_factor, _ϕs)
            ϕ_i += coeff_val * ΔE_i
            return ϕ_to_z(ϕ_i, _rf_factor, _ϕs)
        end
        
        # Process all particles
        for i in 1:length(particles)
            z_values[i] = StochasticAD.propagate(
                (_rf, _ϕs) -> phase_fn(_rf, _ϕs, particles.coordinates.z[i], coeff, particles.coordinates.ΔE[i]),
                rf_factor, ϕs
            )
        end
    else
        # Standard implementation - vectorized calculation
        coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
        
        # Process all particles
        for i in 1:length(particles)
            ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
            ϕ_i += coeff * particles.coordinates.ΔE[i]
            particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
        end
    end
end