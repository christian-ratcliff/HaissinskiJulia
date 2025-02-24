begin
    using Distributions
    using Random
    Random.seed!(12345)
    using LaTeXStrings
    using Plots
    using StatsPlots
    using StochasticAD
    using LoopVectorization
    using ProgressMeter
    using FHist
    using FFTW
    using Interpolations
end




begin
    include("data_structures.jl")
    include("utils.jl")
end


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

    particles = StructArray(
    coordinates = StructArray(Coordinate.(z_vals, ΔE_vals)),  # Ensure StructArray!
    uncertainty = StructArray(Coordinate.(zeros(num_particles), zeros(num_particles))),
    derivative = StructArray(Coordinate.(zeros(num_particles), zeros(num_particles))),
    derivative_uncertainty = StructArray(Coordinate.(zeros(num_particles), zeros(num_particles))))

    return particles, σ_E, σ_z, energy
end

function BeamTurn(n_turns::Integer, n_particles::Integer)
    T = Float64  # Explicit type
    N = n_turns + 1  # Number of stored turns

    GC.enable(false)  # Disable GC for performance
    try
        total_size = n_particles * N

        # Preallocate large contiguous arrays
        z_data = Vector{T}(undef, total_size)
        ΔE_data = Vector{T}(undef, total_size)


        # Preallocate arrays for uncertainties and derivatives
        uncertainty_z = Vector{T}(undef, total_size)
        uncertainty_ΔE = Vector{T}(undef, total_size)

        derivative_z = Vector{T}(undef, total_size)
        derivative_ΔE = Vector{T}(undef, total_size)


        derivative_uncertainty_z = Vector{T}(undef, total_size)
        derivative_uncertainty_ΔE = Vector{T}(undef, total_size)


        # Preallocate states
        states = Vector{StructArray{Particle{T}}}(undef, N)

        @inbounds for i in 1:N
            idx_range = ((i-1) * n_particles + 1):(i * n_particles)

            states[i] = StructArray(
                coordinates = StructArray(Coordinate{T}.(z_data[idx_range], 
                                                         ΔE_data[idx_range])),
                uncertainty = StructArray(Coordinate{T}.(uncertainty_z[idx_range], 
                                                         uncertainty_ΔE[idx_range])),
                derivative = StructArray(Coordinate{T}.(derivative_z[idx_range], 
                                                        derivative_ΔE[idx_range])),
                derivative_uncertainty = StructArray(Coordinate{T}.(derivative_uncertainty_z[idx_range], 
                                                                     derivative_uncertainty_ΔE[idx_range]))
            )
        end

        return BeamTurn{T, N}(states)
    finally
        GC.enable(true)  # Re-enable GC
    end
end

function apply_wakefield_inplace!(
    particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64},#::StructArray{Particle{T}}, 
    buffers::SimulationBuffers{T}, 
    wake_factor::T, 
    wake_sqrt::T, 
    cτ::T,  
    n_particles::Int,
    current::T,
    σ_z::T,
    bin_edges::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}
    ) where T<:Float64
    
    
    if !all(iszero, buffers.λ)
        fill!(buffers.λ, zero(T))::Vector{T}
    end
    if !all(iszero, buffers.WF_temp)
        fill!(buffers.WF_temp, zero(T))::Vector{T}
    end
    if !all(iszero, buffers.convol)
        fill!(buffers.convol, zero(T))::Vector{Complex{T}}
    end


    
    inv_cτ::Float64 = 1 / cτ

    # Calculate histogram
    bin_centers::Vector{T}, bin_amounts::Vector{T} = calculate_histogram(particles.coordinates.z, bin_edges)
    nbins::Int64 = length(bin_centers)
    power_2_length::Int64 = nbins  * 2 #next_power_of_two(2*nbins-1)
    
    
    # Calculate wake function for each bin
    for i in eachindex(bin_centers)
        z= bin_centers[i]
        buffers.WF_temp[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    # Calculate line charge density using Gaussian smoothing
    delta_std::Float64 = (15 * σ_z) / σ_z / 100
    @turbo for i in eachindex(bin_centers)
        buffers.λ[i] = delta(bin_centers[i], delta_std)
    end
    
    # Prepare arrays for convolution
    normalized_amounts::Vector{Float64} = bin_amounts .* (1/n_particles)
    λ = buffers.λ[1:nbins]
    WF_temp = buffers.WF_temp[1:nbins]
    convol = buffers.convol[1:power_2_length]
    
    # Perform convolution and scale by current
    convol .= FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length) .* current
    
    # Interpolate results back to particle positions
    temp_z = range(minimum(particles.coordinates.z), maximum(particles.coordinates.z), length=length(convol))
    resize!(buffers.potential, length(particles.coordinates.z))
    buffers.potential .= LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line()).(particles.coordinates.z)
    
    # Update particle energies and calculate wake function
    @turbo for i in eachindex(particles.coordinates.z)
        z = particles.coordinates.z[i]
        particles.coordinates.ΔE[i] -= buffers.potential[i]
        buffers.WF[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    return nothing
end

function longitudinal_evolve!(
    n_turns::Int,
    particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64},#::StructArrays.StructVector{B, U, Float64},
    ϕs::T,
    α_c::T,
    mass::T,
    voltage::T,
    harmonic::Int64,
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
    plot_WF::Bool=false)::Tuple where {B <: NamedTuple, U <: NamedTuple, T <: Float64}

    # Pre-compute physical constants
    γ0::Float64 = E0 / mass
    β0::Float64 = sqrt(1 - 1/γ0^2)
    η0::Float64 = α_c - 1/(γ0^2)
    sin_ϕs::Float64 = sin(ϕs)
    rf_factor::Float64 = freq_rf * 2π / (β0 * SPEED_LIGHT)
    σ_E0::Float64 = std(particles.coordinates.ΔE)
    σ_z0::Float64 = std(particles.coordinates.z)

    # Initialize buffers
    n_particles::Int64 = length(particles.coordinates.z)
    buffers = create_simulation_buffers(n_particles, Int(n_particles/10), T)
    nbins::Int64 = next_power_of_two(Int(10^(ceil(Int, log10(n_particles)-2))))
    bin_edges = range(-7.5*σ_z, 7.5*σ_z, length=nbins+1)

    # Initialize wakefield parameters if needed
    if use_wakefield
        kp::Float64 = T(3e1)
        Z0::Float64 = T(120π)
        cτ::Float64 = T(4e-3)
        wake_factor::Float64 = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt::Float64 = sqrt(2*kp/pipe_radius)
    end

    # Setup progress meter
    if display_counter
        p::Progress = Progress(n_turns, desc="Simulating Turns: ")
    end


    # Main evolution loop
    @inbounds for turn in 1:n_turns

        # RF voltage kick
        σ_E::Float64 = std(particles.coordinates.ΔE)
        σ_z::Float64 = std(particles.coordinates.z)

        energy_gain_from_voltage!(voltage, sin_ϕs, rf_factor, ϕs, n_particles, particles)

        if use_excitation
            quantum_excitation!(E0, acc_radius, σ_E0, buffers.potential, n_particles, particles)
        end

        if SR_damping
            synchrotron_radiation!(E0, acc_radius, n_particles, particles)
        end

        if use_wakefield
            curr::Float64  =  (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles /E0 /2/π/acc_radius * σ_z / (η0*σ_E0^2)
            apply_wakefield_inplace!(
                    particles, buffers, wake_factor, wake_sqrt, cτ,
                    n_particles, curr, σ_z, bin_edges
                )
        end

        if update_E0
            E0 += voltage * sin_ϕs
            γ0= E0/mass 
            β0= sqrt(1 - 1/γ0^2)
            if SR_damping
                ∂U_∂E::Float64 = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
                E0 -= ∂U_∂E * E0  / 4
                γ0 = E0/mass 
                β0= sqrt(1 - 1/γ0^2)
            end
            if use_wakefield
                E0 += mean(particles.coordinates.ΔE)
                particles.coordinates.ΔE .-= mean(particles.coordinates.ΔE)
            end
        end

        if update_η
            @turbo for i in 1:n_particles
                buffers.Δγ[i] = particles.coordinates.ΔE[i] / mass
                buffers.η[i] = α_c - 1/(γ0 + buffers.Δγ[i])^2
                buffers.coeff[i] = 2π * harmonic * buffers.η[i] / (β0 * β0 * E0)
                buffers.ϕ[i] = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
                buffers.ϕ[i] += buffers.coeff[i] * particles.coordinates.ΔE[i]
            end
        else
            coeff::Float64 = 2π * harmonic * η0 / (β0 * β0 * E0)
            @turbo for i in 1:n_particles
                buffers.ϕ[i] = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
                buffers.ϕ[i] += coeff * particles.coordinates.ΔE[i]
            end
        end

        rf_factor = calc_rf_factor(freq_rf, β0)
        ϕ_to_z(buffers.ϕ, rf_factor, ϕs, n_particles, particles)

        if display_counter
            next!(p)
        end
    end
    return (σ_E, σ_z, E0)
end



function energy_gain_from_voltage!(
    voltage::T,
    sin_ϕs::T,
    rf_factor::T,
    ϕs::T,
    n_particles::Int64,
    particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64}#::StructArray{Particle{T}}
    ) where T<:Float64

    @turbo for i in 1:n_particles
        particles.coordinates.ΔE[i] += voltage * (sin(z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)) - sin_ϕs)
    end
end

function quantum_excitation!(E0::T, acc_radius::T, σ_E0::T, buffer::Vector{Float64}, n_particles::Int64, particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64}#::StructArray{Particle{T}}
    ) where T<:Float64
    ∂U_∂E::Float64 = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
    excitation::Float64 = sqrt(1-(1-∂U_∂E)^2) * σ_E0
    
    samples = zeros(Float64, 100)
    randn!(buffer)::Vector{Float64}
    X(p::StochasticTriple, buff::Float64, excitation::Float64) = p + excitation * buff
    for i in 1:n_particles
        # particles.coordinates.ΔE[i] = particles.coordinates.ΔE[i] + excitation * buffer[i] #This is not actually the potential, merely a random number with the right distribution, I just use the buffer because its already allocated
        
        samples .= [derivative_estimate(p -> X(p, buffer[i], excitation), particles.coordinates.ΔE[i]) for j in 1:100]
        particles.derivative.ΔE[i] = mean(samples)
        particles.coordinates.ΔE[i] = particles.coordinates.ΔE[i] + excitation * buffer[i]
        # println(mean(samples))
        particles.derivative_uncertainty.ΔE[i] = std(samples) / sqrt(1000)
    end
    
end

function synchrotron_radiation!(E0::T, acc_radius::T, n_particles::Int64, particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64}#::StructArray{Particle{T}}
    ) where T<:Float64
    ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
    @turbo for i in 1:n_particles
        particles.coordinates.ΔE[i] *= (1 - ∂U_∂E)
    end
end

