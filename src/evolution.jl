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
        params::SimulationParameters{T},
        buffers::SimulationBuffers{T}
    ) where T<:Float64 -> Tuple{T, T, T}

Evolve particles through the accelerator for multiple turns.

Returns: (σ_E, σ_z, E0) - Final energy spread, bunch length, and reference energy
"""
function longitudinal_evolve!(
    particles::StructArray{Particle{T}},
    params::SimulationParameters{T},
    buffers::SimulationBuffers{T}
) where T<:Float64
    
    # Extract parameters
    E0::T = params.E0
    mass::T = params.mass
    voltage::T = params.voltage
    harmonic::Int = params.harmonic
    radius::T = params.radius
    freq_rf::T = params.freq_rf
    pipe_radius::T = params.pipe_radius
    α_c::T = params.α_c
    ϕs::T = params.ϕs
    n_turns::Int = params.n_turns
    use_wakefield::Bool = params.use_wakefield
    update_η::Bool = params.update_η
    update_E0::Bool = params.update_E0
    SR_damping::Bool = params.SR_damping
    use_excitation::Bool = params.use_excitation
    
    # Pre-compute physical constants
    γ0::T = E0 / mass
    β0::T = sqrt(1 - 1/γ0^2)
    η0::T = α_c - 1/(γ0^2)
    sin_ϕs::T = sin(ϕs)
    rf_factor::T = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Get initial spreads
    n_particles::Int = length(particles)
    σ_E0::T = std(particles.coordinates.ΔE)
    σ_z0::T = std(particles.coordinates.z)
    
    if use_wakefield
        nbins::Int = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
        bin_edges = range(-7.5*σ_z0, 7.5*σ_z0, length=nbins+1)
        kp::T = 3e1
        Z0::T = 120π
        cτ::T = 4e-3
        wake_factor::T = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt::T = sqrt(2*kp/pipe_radius)
    end
    
    # Setup progress meter
    p = Progress(n_turns, desc="Simulating Turns: ")
    
    # # For tracking convergence
    # σ_E_buffer = CircularBuffer{T}(50)
    # E0_buffer = CircularBuffer{T}(50)
    # σ_z_buffer = CircularBuffer{T}(50)
    # push!(σ_E_buffer, σ_E0)
    # push!(E0_buffer, E0)
    # push!(σ_z_buffer, σ_z0)
    
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
            curr::T = (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles /E0 /2/π/radius * σ_z / (η0*σ_E0^2)
            apply_wakefield_inplace!(particles, buffers, wake_factor, wake_sqrt, cτ, curr, σ_z, bin_edges)
            
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
                ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
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
                coeff_i = 2π * harmonic * η_i / (β0 * β0 * E0)
                
                # Get current phase
                ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
                
                # Update phase
                ϕ_i += coeff_i * particles.coordinates.ΔE[i]
                
                # Update longitudinal position
                particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
            end
        else
            # Using constant slip factor
            coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
            
            for i in 1:n_particles
                # Get current phase
                ϕ_i = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
                
                # Update phase
                ϕ_i += coeff * particles.coordinates.ΔE[i]
                
                # Update longitudinal position
                particles.coordinates.z[i] = ϕ_to_z(ϕ_i, rf_factor, ϕs)
            end
        end
        
        # Update RF factor with new beta
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        
        # Update tracking buffers
        # push!(σ_E_buffer, σ_E)
        # push!(E0_buffer, E0)
        # push!(σ_z_buffer, σ_z)
        
        # Check for convergence
        # if abs(mean(σ_E_buffer)/mean(E0_buffer) - σ_E/E0) < 1e-9
        #     @info "Converged at turn $turn with σ_E = $(mean(σ_E_buffer))"
        #     σ_E = mean(σ_E_buffer)
        #     σ_z = mean(σ_z_buffer)
        #     E0 = mean(E0_buffer)
        #     return σ_E, σ_z, E0
        # end
        
        # Update progress
        next!(p)
    end
    
    # Final spreads if not converged
    # particles.coordinates.z = [p.coordinates.z for p in particles]
    # particles.coordinates.ΔE = [p.coordinates.ΔE for p in particles]
    σ_E = std(particles.coordinates.ΔE)
    σ_z = std(particles.coordinates.z)
    
    return σ_E, σ_z, E0
end