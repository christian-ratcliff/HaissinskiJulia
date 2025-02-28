"""
utils.jl - Utility functions for beam simulations

This file contains common utility functions used throughout the simulation:
- Field copying and assignment
- FFT and convolution operations
- Buffer management
- Histogram calculations
- Phase space transformations
"""

using LoopVectorization
using FFTW
using FHist
using Interpolations

"""
    threaded_fieldwise_copy!(destination, source)

Copy particle fields from source to destination in a thread-safe manner.
"""
function threaded_fieldwise_copy!(destination::StructArray{Particle{T}}, source::StructArray{Particle{T}}) where T<:Float64
    @assert length(destination) == length(source)
    Threads.@threads for i in 1:length(source)
        destination[i] = Particle(
            Coordinate(source.coordinates.z[i], source.coordinates.ΔE[i])
        )
    end
end

"""
    assign_to_turn!(particle_trajectory, particle_states, turn)

Assign current particle states to the specified turn in the trajectory.
"""
function assign_to_turn!(particle_trajectory::BeamTurn{T}, particle_states::StructArray{Particle{T}}, turn::Integer) where T<:Float64
    threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
end

"""
    delta(x::T, σ::T) where T<:Float64 -> T

Calculate a Gaussian delta function for beam distribution smoothing.
"""
function delta(x::T, σ::T)::T where T<:Float64
    σ_inv = 1 / (sqrt(2 * π) * σ)
    exp_factor = -0.5 / (σ^2)
    return σ_inv * exp(x^2 * exp_factor)
end

"""
    FastConv1D(f::AbstractVector{T}, g::AbstractVector{T}) where T -> Vector{Complex{T}}

Compute the fast convolution of two vectors using FFT.
"""
function FastConv1D(f::AbstractVector{T}, g::AbstractVector{T})::Vector{Complex{T}} where T<:Float64
    return ifft(fft(f) .* fft(g))
end

"""
    FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int) where T

Compute linear convolution with automatic padding to power of 2 length.
"""
function FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int64)::Vector{Complex{T}} where T<:Float64
    pad_and_ensure_power_of_two!(f, g, power_2_length)
    return FastConv1D(f, g)
end

"""
    is_power_of_two(n::Int) -> Bool

Check if a number is a power of two using bitwise operations.
"""
function is_power_of_two(n::Int64)::Bool
    return (n & (n - 1)) == 0 && n > 0
end

"""
    next_power_of_two(n::Int) -> Int

Find the next power of two greater than or equal to n.
"""
function next_power_of_two(n::Int64)::Int64
    return Int64(2^(ceil(log2(n))))
end

"""
    create_simulation_buffers(n_particles::Int, nbins::Int, T::Type=Float64) -> SimulationBuffers{T}

Create pre-allocated buffers for efficient simulation calculations.
"""
function create_simulation_buffers(n_particles::Int64, nbins::Int64, T::Type=Float64)
    # Pre-allocate all vectors in parallel groups based on size
    particle_vectors = Vector{Vector{T}}(undef, 9)  # For n_particles sized vectors
    bin_vectors = Vector{Vector{T}}(undef, 2)       # For nbins sized vectors
    
    # Initialize n_particles sized vectors in parallel
    Threads.@threads for i in 1:9
        particle_vectors[i] = Vector{T}(undef, n_particles)
    end
    
    # Initialize nbins sized vectors in parallel
    Threads.@threads for i in 1:2
        bin_vectors[i] = Vector{T}(undef, nbins)
    end
    
    # Complex vector (single allocation)
    complex_vector = Vector{Complex{T}}(undef, nbins)
    
    # Random buffer
    random_buffer = Vector{T}(undef, n_particles)
    
    SimulationBuffers{T}(
        particle_vectors[1],   # WF
        particle_vectors[2],   # potential
        particle_vectors[3],   # Δγ
        particle_vectors[4],   # η
        particle_vectors[5],   # coeff
        particle_vectors[6],   # temp_z
        particle_vectors[7],   # temp_ΔE
        particle_vectors[8],   # temp_ϕ
        bin_vectors[1],        # WF_temp
        bin_vectors[2],        # λ
        complex_vector,        # convol
        particle_vectors[9],   # ϕ
        random_buffer          # random_buffer
    )
end

"""
    pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T -> Nothing

Pad vectors to power-of-two length for efficient FFT operations.
"""
function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T<:Float64
    N::Int64 = length(f)
    M::Int64 = length(g)
    
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
    calculate_histogram(data::Vector{T}, bins_edges) -> Tuple{Vector{T}, Vector{Int}}

Calculate histogram of data with specified bin edges.
"""
function calculate_histogram(data::Vector{T}, bins_edges) where T<:Float64
    histo = Hist1D(data, binedges=bins_edges)
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end

"""
    z_to_ϕ(z_val::T, rf_factor::T, ϕs::T) where T<:Float64 -> T

Convert longitudinal position to RF phase.
"""
function z_to_ϕ(z_val::T, rf_factor::T, ϕs::T) where T<:Float64
    return -(z_val * rf_factor - ϕs)
end

"""
    ϕ_to_z(ϕ_val::T, rf_factor::T, ϕs::T) where T<:Float64 -> T

Convert RF phase to longitudinal position.
"""
function ϕ_to_z(ϕ_val::T, rf_factor::T, ϕs::T) where T<:Float64
    return (-ϕ_val + ϕs) / rf_factor
end

"""
    calc_rf_factor(freq_rf::T, β::T) where T<:Float64 -> T

Calculate RF factor from RF frequency and relativistic beta.
"""
function calc_rf_factor(freq_rf::T, β::T) where T<:Float64
    return freq_rf * 2π / (β * 299792458.0)  
end