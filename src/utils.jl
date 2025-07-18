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
@inline function delta(x::T, σ::T)::T where T<:Float64
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
# function create_simulation_buffers(n_particles::Int64, nbins::Int64, T::Type=Float64)
#     # Pre-allocate all vectors in parallel groups based on size
#     particle_vectors = Vector{Vector{T}}(undef, 9)  # For n_particles sized vectors
#     bin_vectors = Vector{Vector{T}}(undef, 2)       # For nbins sized vectors
    
#     # Initialize n_particles sized vectors in parallel
#     Threads.@threads for i in 1:9
#         particle_vectors[i] = Vector{T}(undef, n_particles)
#     end
    
#     # Initialize nbins sized vectors in parallel
#     Threads.@threads for i in 1:2
#         bin_vectors[i] = Vector{T}(undef, nbins)
#     end
    
#     # Complex vector (single allocation)
#     complex_vector = Vector{Complex{T}}(undef, nbins)
    
#     # Random buffer
#     random_buffer = Vector{T}(undef, n_particles)
    
#     SimulationBuffers{T}(
#         particle_vectors[1],   # WF
#         particle_vectors[2],   # potential
#         particle_vectors[3],   # Δγ
#         particle_vectors[4],   # η
#         particle_vectors[5],   # coeff
#         particle_vectors[6],   # temp_z
#         particle_vectors[7],   # temp_ΔE
#         particle_vectors[8],   # temp_ϕ
#         bin_vectors[1],        # WF_temp
#         bin_vectors[2],        # λ
#         complex_vector,        # convol
#         particle_vectors[9],   # ϕ
#         random_buffer          # random_buffer
#     )
# end

function create_simulation_buffers(n_particles::Int64, nbins::Int64, T::Type=StochasticTriple)
    # Create power-of-two sized buffers for FFT
    power_2_length = next_power_of_two(nbins * 2)
    
    # Pre-allocate vectors in groups based on size for better cache locality
    particle_vectors = Vector{Vector{T}}(undef, 9)
    bin_vectors = Vector{Vector{T}}(undef, 3)
    complex_vectors = Vector{Vector{Complex{T}}}(undef, 3)
    
    # Initialize in parallel
    Threads.@threads for i in 1:9
        particle_vectors[i] = Vector{T}(undef, n_particles)
    end
    
    Threads.@threads for i in 1:3
        bin_vectors[i] = Vector{T}(undef, nbins)
    end
    
    Threads.@threads for i in 1:3
        complex_vectors[i] = Vector{Complex{T}}(undef, power_2_length)
    end
    
    # Integer buffer for histogram
    bin_counts = Vector{Int}(undef, nbins)
    
    # Random buffer
    random_buffer = Vector{T}(undef, n_particles)
    
    # Real part buffer
    real_buffer = Vector{T}(undef, power_2_length)
    
    # Create thread-local storage for parallel operations
    n_threads = Threads.nthreads()
    # thread_local_buffers = Vector{Dict{Symbol, Any}}(undef, n_threads)
    
    # Initialize thread-local buffers
    # for i in 1:n_threads
    #     thread_local_buffers[i] = Dict{Symbol, Any}(
    #         :sum => zero(T),
    #         :count => 0,
    #         :bin_counts => zeros(T, nbins),
    #         :temp_array => Vector{T}(undef, n_particles ÷ n_threads + 1)
    #     )
    # end
    
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
        complex_vectors[1],    # convol
        particle_vectors[9],   # ϕ
        random_buffer,         # random_buffer
        bin_vectors[3],        # normalized_λ
        complex_vectors[2],    # fft_buffer1
        complex_vectors[3],    # fft_buffer2
        real_buffer,           # real_buffer
        bin_counts,            # bin_counts
        # thread_local_buffers   # thread_local_buffers
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

function calculate_histogram!(
    data::Vector{T}, 
    bin_edges::AbstractRange, 
    bin_counts::Vector{Int}
) where T<:Float64
    fill!(bin_counts, 0)
    nbins = length(bin_counts)
    
    # Count occurrences in each bin
    @inbounds for val in data
        bin_idx = searchsortedfirst(bin_edges, val) - 1
        if 1 <= bin_idx <= nbins
            bin_counts[bin_idx] += 1
        end
    end
    
    return nothing
end

function in_place_convolution!(
    result::Vector{Complex{T}},
    f::AbstractVector{T},
    g::AbstractVector{T},
    power_2_length::Int,
    fft_buffer1::Vector{Complex{T}},
    fft_buffer2::Vector{Complex{T}}
) where T<:Float64
    
    n_f = length(f)
    n_g = length(g)
    
    # Prepare input buffers
    fill!(fft_buffer1, zero(Complex{T}))
    fill!(fft_buffer2, zero(Complex{T}))
    
    # Copy data to fft buffers
    @inbounds for i in 1:n_f
        fft_buffer1[i] = Complex{T}(f[i])
    end
    
    @inbounds for i in 1:n_g
        fft_buffer2[i] = Complex{T}(g[i])
    end
    
    # In-place FFT
    plan_fft = plan_fft!(fft_buffer1)
    plan_fft * fft_buffer1
    plan_fft * fft_buffer2
    
    # Element-wise multiplication
    @inbounds for i in 1:power_2_length
        fft_buffer1[i] *= fft_buffer2[i]
    end
    
    # In-place IFFT
    plan_ifft = plan_ifft!(fft_buffer1)
    plan_ifft * fft_buffer1
    
    # Copy to result
    copyto!(result, fft_buffer1)
    
    return nothing
end

"""
    z_to_ϕ(z_val, rf_factor, ϕs) -> Any

Convert longitudinal position to RF phase.
Compatible with StochasticTriple.
"""
@inline function z_to_ϕ(z_val, rf_factor, ϕs)
    return -(z_val * rf_factor - ϕs)
end

"""
    ϕ_to_z(ϕ_val, rf_factor, ϕs) -> Any

Convert RF phase to longitudinal position.
Compatible with StochasticTriple.
"""
@inline function ϕ_to_z(ϕ_val, rf_factor, ϕs)
    return (-ϕ_val + ϕs) / rf_factor
end

"""
    calc_rf_factor(freq_rf::T, β::T) where T<:Float64 -> T

Calculate RF factor from RF frequency and relativistic beta.
"""
function calc_rf_factor(freq_rf::T, β::T) where T<:Float64
    return freq_rf * 2π / (β * 299792458.0)  
end

"""
    copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64

Efficiently copy particle data without allocations.
"""
function copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64
    @assert length(dst) == length(src)
    copyto!(dst.coordinates.z, src.coordinates.z)
    copyto!(dst.coordinates.ΔE, src.coordinates.ΔE)
    if hasproperty(dst, :uncertainty) && hasproperty(src, :uncertainty)
        copyto!(dst.uncertainty.z, src.uncertainty.z)
        copyto!(dst.uncertainty.ΔE, src.uncertainty.ΔE)
    end
    return dst
end


@inline function compute_std(x::AbstractVector{T}) where T<:Float64
    n = length(x)
    μ = compute_mean(x)
    s = zero(T)
    @turbo for i in 1:n
        diff = x[i] - μ
        s += diff * diff
    end
    return sqrt(s / (n - 1))
end

@inline function compute_mean_diff(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Float64
    s = zero(T)
    n = length(x)
    @turbo for i in 1:n
        s += x[i] - y[i]
    end
    return s / n
end

function subtract_mean_inplace!(x::AbstractVector{T}, mean_val) where T<:Float64
    @turbo for i in 1:length(x)
        x[i] -= mean_val
    end
end