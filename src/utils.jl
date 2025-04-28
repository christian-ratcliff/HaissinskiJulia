# src/utils.jl
"""
utils.jl - Utility functions for beam simulations

Contains common utility functions supporting both serial and MPI modes:
- Particle data copying.
- FFT and convolution operations.
- Buffer management adaptable to serial/MPI.
- Histogram calculations (local and preparation for global).
- Phase space transformations.
- Basic statistical helpers (local and global via MPI).
"""

using LoopVectorization
using FFTW
using FHist
using Interpolations
using MPI

"""
    threaded_fieldwise_copy!(destination, source)

Copy particle fields from source to destination using threads.
"""
function threaded_fieldwise_copy!(destination::StructArray{Particle{T}}, source::StructArray{Particle{T}}) where T<:Float64
    @assert length(destination) == length(source) "Destination and source particle arrays must have the same length."
    # Ensure Coordinate struct exists and is accessible
    coord_type = typeof(destination.coordinates[1])
    Threads.@threads for i in 1:length(source)
        # Direct assignment is efficient for StructArrays if types match exactly
        # Creating new Particle/Coordinate objects can be less efficient
        destination.coordinates.z[i] = source.coordinates.z[i]
        destination.coordinates.ΔE[i] = source.coordinates.ΔE[i]
    end
end

"""
    assign_to_turn!(particle_trajectory, particle_states, turn)

Assign current particle states to the specified turn in the trajectory.
"""
function assign_to_turn!(particle_trajectory::BeamTurn{T}, particle_states::StructArray{Particle{T}}, turn::Integer) where T<:Float64
    if turn < 1 || turn > length(particle_trajectory.states)
        error("Invalid turn number: $turn. Must be between 1 and $(length(particle_trajectory.states)).")
    end
    # Use efficient copy if target struct already exists
    copyto_particles!(particle_trajectory.states[turn], particle_states)
    # Fallback using threaded copy if needed, but copyto_particles! is preferred
    # threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
end


"""
    delta(x::T, σ::T) where T<:Float64 -> T

Calculate a Gaussian kernel value for smoothing. Avoids singularity at σ=0.
"""
@inline function delta(x::T, σ::T)::T where T<:Float64
    # if σ <= 0
    #     # Return a very narrow spike approximation if sigma is non-positive
    #     return abs(x) < eps(T) ? T(1.0 / (sqrt(2 * π) * eps(T))) : zero(T)
    # end
    σ_inv_sqrt2pi = 1 / (sqrt(2 * π) * σ)
    exp_factor = -0.5 / (σ * σ)
    return σ_inv_sqrt2pi * exp(x * x * exp_factor)
end

"""
    FastConv1D(f::AbstractVector{Complex{T}}, g::AbstractVector{Complex{T}}) where T -> Vector{Complex{T}}

Compute the fast convolution of two complex vectors using FFT (assumes inputs are already FFT'd).
Corrected function signature and operation for clarity. This is usually called after FFTs are performed.
If inputs are real vectors, FFT them first.

Note: The original function description was slightly misleading. This performs element-wise
multiplication in frequency domain and inverse FFT.
"""
function FastConv1D(fft_f::AbstractVector{Complex{T}}, fft_g::AbstractVector{Complex{T}})::Vector{Complex{T}} where T<:Float64
    # Assumes fft_f = fft(f) and fft_g = fft(g) are provided
    # Performs element-wise product and inverse FFT
    return ifft(fft_f .* fft_g)
end

"""
    FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int) where T

Compute linear convolution with automatic padding to power of 2 length.
Uses temporary buffers for FFTs if available, otherwise allocates.
"""
function FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int64)::Vector{Complex{T}} where T<:Float64
    # Simple implementation without buffer reuse for clarity here
    f_padded = Vector{T}(undef, power_2_length)
    g_padded = Vector{T}(undef, power_2_length)
    fill!(f_padded, 0); fill!(g_padded, 0)
    copyto!(f_padded, 1, f, 1, length(f))
    copyto!(g_padded, 1, g, 1, length(g))

    fft_f = fft(f_padded)
    fft_g = fft(g_padded)

    return ifft(fft_f .* fft_g) # Calls ifft internally
end


"""
    is_power_of_two(n::Int) -> Bool

Check if a number is a power of two using bitwise operations. Handles n <= 0.
"""
function is_power_of_two(n::Int64)::Bool
    return n > 0 && (n & (n - 1)) == 0
end

"""
    next_power_of_two(n::Int) -> Int

Find the next power of two greater than or equal to n. Handles n <= 0.
"""
function next_power_of_two(n::Int64)::Int64
    n <= 0 && return 1 # Return 1 for non-positive input
    return Int64(2^ceil(Int, log2(n))) # Use Int for exponent ceiling
end

"""
    create_simulation_buffers(
        n_particles::Int64,
        nbins::Int64,
        use_mpi::Bool,
        comm_size::Int = 1; # Needed for thread local buffer sizing in MPI mode
        T::Type=Float64
    ) -> SimulationBuffers{T}

Create pre-allocated buffers for simulation.
Sizes buffers based on whether MPI is used (local vs global particle counts).

Args:
    n_particles (Int64): Number of particles. In MPI mode, this is n_local_particles.
                         In serial mode, this is the total number of particles.
    nbins (Int64): Number of bins for histograms and related arrays (a global property).
    use_mpi (Bool): Flag indicating if running in MPI mode.
    comm_size (Int): Number of MPI ranks (used for thread buffer sizing in MPI).
    T (Type): Floating point type for buffers.

Returns:
    SimulationBuffers{T}: The allocated buffers structure.
"""
function create_simulation_buffers(
    n_particles::Int64, # Represents n_local in MPI, n_global in serial
    nbins::Int64,
    use_mpi::Bool,
    comm_size::Int = 1; # Default to 1 for serial case
    T::Type=Float64
    )::SimulationBuffers{T}

    if nbins <= 0
        error("Number of bins (nbins) must be positive. Got $nbins.")
    end
    if n_particles < 0
        error("Number of particles must be non-negative. Got $n_particles.")
    end

    # FFT length based on global property nbins
    power_2_length = next_power_of_two(nbins * 2)

    # Allocate particle-sized buffers (size depends on mode)
    _WF = Vector{T}(undef, n_particles)
    _potential = Vector{T}(undef, n_particles)
    _Δγ = Vector{T}(undef, n_particles)
    _η = Vector{T}(undef, n_particles)
    _coeff = Vector{T}(undef, n_particles)
    _temp_z = Vector{T}(undef, n_particles)
    _temp_ΔE = Vector{T}(undef, n_particles)
    _temp_ϕ = Vector{T}(undef, n_particles)
    _ϕ = Vector{T}(undef, n_particles)
    _random_buffer = Vector{T}(undef, n_particles)
    _ΔE_initial_turn = Vector{T}(undef, n_particles)
    _mpi_fft_W = Vector{Complex{T}}(undef, n_particles)
    _mpi_fft_L = Vector{Complex{T}}(undef, n_particles)
    _mpi_convol_freq = Vector{Complex{T}}(undef, n_particles)

    # Allocate bin-sized buffers (global property)
    _WF_temp = Vector{T}(undef, nbins)
    _λ = Vector{T}(undef, nbins)
    _normalized_λ = Vector{T}(undef, nbins)
    _bin_counts = Vector{Int}(undef, nbins) # Local counts

    # Allocate FFT-sized buffers (global property)
    _convol = Vector{Complex{T}}(undef, power_2_length)
    _fft_buffer1 = Vector{Complex{T}}(undef, power_2_length)
    _fft_buffer2 = Vector{Complex{T}}(undef, power_2_length)
    _real_buffer = Vector{T}(undef, power_2_length)

    # Create and store FFT plans 
    _fft_plans = Dict{Symbol, Any}()
    _fft_plans[:fft_plan1] = plan_fft!(_fft_buffer1; flags=FFTW.PATIENT)
    _fft_plans[:fft_plan2] = plan_fft!(_fft_buffer2; flags=FFTW.PATIENT) 
    _fft_plans[:ifft_plan] = plan_ifft!(_convol; flags=FFTW.PATIENT)

    _interp_indices = Vector{Int}(undef, n_particles)
    _interp_weights = Vector{T}(undef, n_particles)

    # Allocate MPI-specific buffers (global property)
    # These are allocated even in serial mode but will be unused.
    _global_bin_counts = Vector{Int}(undef, nbins)
    _normalized_global_amounts = Vector{T}(undef, nbins)
    _potential_values_at_centers_global = Vector{T}(undef, nbins)

    # Thread-local storage (useful for serial or threaded loops within MPI rank)
    n_threads = Threads.nthreads()
    _thread_local_buffers = Vector{Dict{Symbol, Any}}(undef, n_threads)
    # Estimate particle chunk size per thread
    # In MPI, n_particles is n_local. In serial, it's n_global.
    # We size based on n_particles / n_threads.
    local_chunk_size_per_thread = max(1, n_particles ÷ n_threads)
    for i in 1:n_threads
        _thread_local_buffers[i] = Dict{Symbol, Any}(
            # Example: size based on particles handled by this rank
             :temp_array => Vector{T}(undef, local_chunk_size_per_thread)
        )
    end

    _scatterv_displs = Vector{Int}(undef, comm_size)
    _scatterv_counts = Vector{Int}(undef, comm_size)

    # Pre-allocate buffers for common MPI Allreduce operations
    _allreduce_energy = Vector{T}(undef, 2)  # For [sum_dE, n_local]
    _allreduce_stats = Vector{T}(undef, 3)   # For [sum, sum_sq, count]
    _allreduce_single = Vector{T}(undef, 1)  # For single value reductions
    
    # Initialize MPI buffers dictionary
    _mpi_buffers = Dict{Symbol, Any}()

    n_threads = Threads.nthreads()
    chunk_size = max(1, n_particles ÷ n_threads)
    _thread_chunks = Vector{UnitRange{Int}}(undef, n_threads)
    
    # Pre-compute chunk ranges for each thread
    for i in 1:n_threads
        chunk_start = (i-1) * chunk_size + 1
        chunk_end = min(i * chunk_size, n_particles)
        _thread_chunks[i] = chunk_start:chunk_end
    end

    return SimulationBuffers{T}(
        _WF, _potential, _Δγ, _η, _coeff, _temp_z, _temp_ΔE, _temp_ϕ, _ϕ, _random_buffer,_ΔE_initial_turn, _mpi_fft_W, _mpi_fft_L, _mpi_convol_freq,
        _WF_temp, _λ, _normalized_λ, _bin_counts,
        _convol, _fft_buffer1, _fft_buffer2, _real_buffer,
        _thread_local_buffers,
        _global_bin_counts,_normalized_global_amounts, _potential_values_at_centers_global, _fft_plans, _interp_indices, _interp_weights,
        _scatterv_counts, _scatterv_displs,
        _mpi_buffers, _allreduce_energy, _allreduce_stats, _allreduce_single,
        _thread_chunks
    )
end


"""
    pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T -> Nothing

Deprecated in favor of direct padding within convolution functions or using FFT plans.
This function resizes vectors, which can be inefficient.
Kept for compatibility if directly called, but recommend avoiding.
"""
function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T<:Float64
    N::Int = length(f)
    M::Int = length(g)

    if length(f) < power_two_length
        f_orig = copy(f)
        resize!(f, power_two_length)
        fill!(view(f, N+1:power_two_length), zero(T))
        copyto!(f, 1, f_orig, 1, N) # Copy back original data
    elseif length(f) > power_two_length
        resize!(f, power_two_length) # Truncate if too long
    end

    if length(g) < power_two_length
        g_orig = copy(g)
        resize!(g, power_two_length)
        fill!(view(g, M+1:power_two_length), zero(T))
        copyto!(g, 1, g_orig, 1, M)
    elseif length(g) > power_two_length
        resize!(g, power_two_length)
    end

    return nothing
end


"""
    calculate_histogram(data::Vector{T}, bins_edges) -> Tuple{Vector{T}, Vector{Int}}

Calculate histogram using FHist. Returns bin centers and counts.
This is primarily used by the serial wakefield calculation.
"""
function calculate_histogram(data::Vector{T}, bins_edges::AbstractRange{T}) where T<:Float64
    if isempty(data)
        nbins = length(bins_edges) - 1
        centers = (bins_edges[1:end-1] .+ bins_edges[2:end]) ./ 2
        return collect(centers), zeros(Int, nbins)
    end
    # Ensure bin edges type matches data type potentially
    histo = Hist1D(data; binedges=bins_edges) # Use keyword argument for clarity
    # Calculate centers from the actual edges used by FHist
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end

"""
    calculate_histogram!(...)

Calculate histogram in-place, populating `bin_counts`.
Used by the serial wakefield calculation.
"""
function calculate_histogram!(
    data::AbstractVector{T},
    bin_edges::AbstractRange,
    bin_counts::Vector{Int} # Output buffer
) where T<:Float64
    fill!(bin_counts, 0) # Reset counts
    nbins = length(bin_counts)
    if length(bin_edges) != nbins + 1
        error("Bin edges length must be nbins + 1. Got $(length(bin_edges)) edges for $nbins bins.")
    end

    # Check edge cases
    if isempty(data); return nothing; end
    first_edge = bin_edges[1]
    last_edge = bin_edges[end]

    @inbounds for val in data
        # Handle values outside the bin range if necessary, here we ignore them
        if val >= first_edge && val < last_edge
            # searchsortedfirst finds the index `k` such that bin_edges[k] >= val
            # The correct bin index is `k-1`
            bin_idx = searchsortedfirst(bin_edges, val) - 1
             # Ensure index is within valid range [1, nbins]
            if bin_idx >= 1 && bin_idx <= nbins
                 bin_counts[bin_idx] += 1
            end
        elseif val == last_edge # Explicitly include the last edge in the last bin
             bin_counts[nbins] += 1
        end
        # Values outside [first_edge, last_edge] are ignored
    end

    return nothing
end

""" Calculate histogram for local data using global bin edges (MPI version helper) """
function calculate_local_histogram!(
    local_data::AbstractVector{T},
    bin_edges::AbstractRange,
    local_bin_counts::Vector{Int} # Output buffer for local counts
) where T<:Float64
    fill!(local_bin_counts, 0)
    nbins = length(local_bin_counts)
    # This function is identical in implementation to calculate_histogram!
    # Kept separate name for clarity in MPI wakefield code path.
    # --- Assertions and Setup ---
    if length(bin_edges) != nbins + 1
        error("Mismatch between bin_edges length ($(length(bin_edges))) and local_bin_counts length ($nbins)")
    end
    if isempty(local_data); return nothing; end

    # Precompute for uniform bins
    first_edge = first(bin_edges)
    last_edge = last(bin_edges) # Use last() for ranges - more robust
    bin_step = step(bin_edges)

    # Check for zero step to avoid division by zero
    if bin_step == 0 && nbins > 0
         error("Bin edges have zero step size.")
    elseif bin_step == 0 && nbins == 0 # Special case: no bins, nothing to do
        return nothing
    end

    inv_bin_step = T(1.0 / bin_step) # Ensure type matches T
    first_edge_T = T(first_edge)     # Ensure type matches T
    last_edge_T = T(last_edge)       # Ensure type matches T
    nbins_Int = Int(nbins)           # Ensure it's Int for clamp

    # --- Main Loop with @turbo ---
    @turbo for i in 1:length(local_data)
        # 1. Calculate the condition (yields true or false)
        # Use '&' which can sometimes be better for SIMD masks
        in_range = (local_data[i] >= first_edge_T) & (local_data[i] <= last_edge_T)

        # 2. Calculate the potential bin index. Clamp handles edge cases.
        # This calculation happens even if out of range, but the result
        # is only used if in_range is true.
        bin_idx_float = (local_data[i] - first_edge_T) * inv_bin_step
        # Clamp takes Int bounds usually
        bin_idx = clamp(floor(Int, bin_idx_float) + 1, 1, nbins_Int)


        @inbounds local_bin_counts[bin_idx] += in_range
    end

    return nothing
end


"""
    in_place_convolution!(...)

Perform convolution using in-place FFTs with provided buffers.
"""
function in_place_convolution!(
    result::Vector{Complex{T}}, # Output buffer (usually buffers.convol)
    f::AbstractVector{T},       # Input signal 1 (e.g., wake function at centers)
    g::AbstractVector{T},       # Input signal 2 (e.g., smoothed lambda)
    power_2_length::Int,
    fft_buffer1::Vector{Complex{T}}, # Workspace buffer 1
    fft_buffer2::Vector{Complex{T}},  # Workspace buffer 2
    buffers::SimulationBuffers{T}
) where T<:Float64

    n_f = length(f)
    n_g = length(g)

    # Ensure buffers have the correct length
    @assert length(fft_buffer1) == power_2_length "fft_buffer1 has incorrect length"
    @assert length(fft_buffer2) == power_2_length "fft_buffer2 has incorrect length"
    @assert length(result) >= power_2_length "Result buffer is too short"

    # Prepare input buffers (copy f and g, zero-padding)
    fill!(fft_buffer1, zero(Complex{T}))
    fill!(fft_buffer2, zero(Complex{T}))

    @inbounds for i in 1:min(n_f, power_2_length)
        fft_buffer1[i] = Complex{T}(f[i])
    end
    @inbounds for i in 1:min(n_g, power_2_length)
        fft_buffer2[i] = Complex{T}(g[i])
    end

    # In-place FFTs
    # Use pre-computed plans instead of creating new ones
    buffers.fft_plans[:fft_plan1] * fft_buffer1
    buffers.fft_plans[:fft_plan2] * fft_buffer2

    # Element-wise multiplication (store result in fft_buffer1)
    @inbounds for i in 1:power_2_length
        fft_buffer1[i] *= fft_buffer2[i]
    end

    # In-place IFFT (result in fft_buffer1)
    buffers.fft_plans[:ifft_plan] * fft_buffer1

    # Copy result to the designated output buffer
    copyto!(result, 1, fft_buffer1, 1, power_2_length)

    return nothing
end

"""
    z_to_ϕ(z_val, rf_factor, ϕs) -> Any

Convert longitudinal position relative to reference particle to RF phase.
"""
@inline function z_to_ϕ(z_val, rf_factor, ϕs)
    return -(z_val * rf_factor - ϕs)
end

"""
    ϕ_to_z(ϕ_val, rf_factor, ϕs) -> Any

Convert RF phase relative to reference particle to longitudinal position.
"""
@inline function ϕ_to_z(ϕ_val, rf_factor, ϕs)
    if rf_factor == 0
        # Handle this case appropriately - maybe return Inf or NaN, or error
        error("RF factor is zero, cannot convert phase to position.")
    end
    return (-ϕ_val + ϕs) / rf_factor
end

"""
    calc_rf_factor(freq_rf::T, β::T) where T<:Float64 -> T

Calculate RF factor ω_rf / (β * c). Handles β=0.
"""
function calc_rf_factor(freq_rf::T, β::T) where T<:Float64
    # Use SPEED_LIGHT constant
    if β <= 0 || SPEED_LIGHT == 0
        # Handle non-physical cases
        return T(Inf) # Or other appropriate value/error
    end
    return freq_rf * (2 * π) / (β * SPEED_LIGHT)
end

"""
    copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64

Efficiently copy particle data between StructArrays without allocations.
Handles potential uncertainty field.
"""
function copyto_particles!(dst::StructArray{Particle{T}}, src::StructArray{Particle{T}}) where T<:Float64
    len = length(src)
    @assert length(dst) == len "Source and destination StructArrays must have the same length."
    if len == 0 return dst end # Handle empty arrays

    # Use internal buffer copy for efficiency
    copyto!(dst.coordinates.z, 1, src.coordinates.z, 1, len)
    copyto!(dst.coordinates.ΔE, 1, src.coordinates.ΔE, 1, len)

    # Check for uncertainty field if it exists
    if hasproperty(dst, :uncertainty) && hasproperty(src, :uncertainty) && hasproperty(dst.uncertainty, :z) && hasproperty(dst.uncertainty, :ΔE)
        copyto!(dst.uncertainty.z, 1, src.uncertainty.z, 1, len)
        copyto!(dst.uncertainty.ΔE, 1, src.uncertainty.ΔE, 1, len)
    end
    return dst
end

# --- Local Statistical Helpers ---

""" Compute mean of a vector efficiently (serial helper). """
@inline function compute_mean(x::AbstractVector{T}) where T<:Float64
    n = length(x)
    if n == 0 return zero(T) end
    s = zero(T)
    # Use LoopVectorization if available and beneficial
    @turbo for i in 1:n
        s += x[i]
    end
    return s / n
end

""" Compute standard deviation of a vector efficiently (serial helper). """
@inline function compute_std(x::AbstractVector{T}) where T<:Float64
    n = length(x)
    if n <= 1 return zero(T) end # Std dev undefined for n=0, 0 for n=1
    μ = compute_mean(x)
    s = zero(T)
    @turbo for i in 1:n
        diff = x[i] - μ
        s += diff * diff
    end
    # Use n-1 for sample standard deviation
    variance = s / (n - 1)
    # Ensure variance is non-negative due to potential floating point errors
    return sqrt(max(zero(T), variance))
end

""" Compute mean difference between two vectors (serial helper). """
@inline function compute_mean_diff(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Float64
    n = length(x)
    @assert n == length(y) "Vectors must have the same length for difference calculation."
    if n == 0 return zero(T) end
    s = zero(T)
    @turbo for i in 1:n
        s += x[i] - y[i]
    end
    return s / n
end

""" Subtract a value from all elements of a vector in-place (serial helper). """
function subtract_mean_inplace!(x::AbstractVector{T}, value_to_subtract) where T<:Float64
    if isempty(x) return end
    @turbo for i in 1:length(x)
        x[i] -= value_to_subtract
    end
end

# --- MPI Statistical Helpers ---

""" Calculate global standard deviation via MPI Allreduce """
function compute_global_std(local_data::AbstractVector{T}, comm::MPI.Comm, buffers::SimulationBuffers{T},) where T<:Float64
    n_local = length(local_data)
    sum_local = zero(T)
    sum_sq_local = zero(T)

    # Calculate local sums efficiently
    if n_local > 0
        @simd for i in 1:n_local
            val = local_data[i]
            sum_local += val
            sum_sq_local += val * val
        end
    end

    # Use pre-allocated buffer instead of creating a new array
    buffers.allreduce_stats[1] = sum_local
    buffers.allreduce_stats[2] = sum_sq_local
    buffers.allreduce_stats[3] = T(n_local)
    MPI.Allreduce!(buffers.allreduce_stats, MPI.SUM, comm)
    
    sum_global = buffers.allreduce_stats[1]
    sum_sq_global = buffers.allreduce_stats[2]
    n_global = Int(round(buffers.allreduce_stats[3]))

    # Calculate global standard deviation
    if n_global <= 1
        return zero(T) # Std dev is 0 for n=1, undefined for n=0
    end
    mean_global = sum_global / n_global
    # Variance = E[X^2] - (E[X])^2 = (sum_sq / n) - mean^2
    variance_global = (sum_sq_global / n_global) - (mean_global * mean_global)
    # Correct for sample standard deviation (factor n / (n - 1))
    # Ensure variance is non-negative before sqrt
    # variance_global = max(zero(T), variance_global) # Already maxed below
    sample_variance_global = max(zero(T), variance_global * n_global / (n_global - 1))

    return sqrt(sample_variance_global)
end


""" Calculate global mean via MPI Allreduce """
function compute_global_mean(local_data::AbstractVector{T}, comm::MPI.Comm) where T<:Float64
     n_local = length(local_data)
     sum_local = zero(T)
     if n_local > 0
         @simd for i in 1:n_local # Use @simd for potential optimization
             sum_local += local_data[i]
         end
     end
     # Reduce local sum and local count
    buffers.allreduce_stats[1] = sum_local
    # buffers.allreduce_stats[2] = sum_sq_local
    buffers.allreduce_stats[3] = T(n_local)
    MPI.Allreduce!(buffers.allreduce_stats, MPI.SUM, comm)
    
    sum_global = buffers.allreduce_stats[1]
    # sum_sq_global = buffers.allreduce_stats[2]
    n_global = Int(round(buffers.allreduce_stats[3]))

     # Calculate global mean, handle n_global = 0 case
     return n_global > 0 ? (sum_global / n_global) : zero(T)
end


"""
    initialize_mpi_buffers!(
        buffers::SimulationBuffers{T},
        comm_size::Int,
        comm::MPI.Comm,
        power_2_length::Int
    ) where T<:Float64

Initialize pre-allocated MPI buffer objects to avoid per-turn allocations.
Should be called once during simulation setup before the main simulation loop.
"""
function initialize_mpi_buffers!(
    buffers::SimulationBuffers{T},
    comm_size::Int,
    comm::MPI.Comm,
    power_2_length::Int
) where T<:Float64
    
    # Get rank
    rank = MPI.Comm_rank(comm)
    
    # Setup counts and displacements - these are already in buffers
    # but we need to populate them
    counts = buffers.scatterv_counts
    displs = buffers.scatterv_displs
    
    # Compute counts and displacements for each rank
    base_chunk_size = power_2_length ÷ comm_size
    remainder = power_2_length % comm_size
    current_displacement = 0
    
    for r in 0:(comm_size-1)
        local_chunk = r < remainder ? base_chunk_size + 1 : base_chunk_size
        counts[r+1] = local_chunk 
        displs[r+1] = current_displacement
        current_displacement += local_chunk
    end
    
    # Get this rank's chunk size
    chunk_size = counts[rank+1]
    
    # Create views for the local chunks
    local_fft_W = view(buffers.mpi_fft_W, 1:chunk_size)
    local_fft_L = view(buffers.mpi_fft_L, 1:chunk_size)
    local_convol_freq = view(buffers.mpi_convol_freq, 1:chunk_size)
    
    # Create MPI buffer objects and store them in the mpi_buffers dictionary
    # These will be reused in each turn
    
    # Receive buffers for all ranks
    buffers.mpi_buffers[:rbuf_W] = MPI.Buffer(local_fft_W)
    buffers.mpi_buffers[:rbuf_L] = MPI.Buffer(local_fft_L)
    buffers.mpi_buffers[:sbuf_convol] = MPI.Buffer(local_convol_freq)
    
    # Send buffers only for rank 0
    if rank == 0
        buffers.mpi_buffers[:sbuf_W] = MPI.VBuffer(buffers.fft_buffer1, counts, displs) 
        buffers.mpi_buffers[:sbuf_L] = MPI.VBuffer(buffers.fft_buffer2, counts, displs)
        buffers.mpi_buffers[:rbuf_convol] = MPI.VBuffer(buffers.convol, counts, displs)
    end
    buffers.mpi_buffers[:initialized] = true
    return nothing
end

function free_mpi_buffer!(buffer)
    if buffer !== nothing
        try
            MPI.free(buffer)
        catch e
            # Handle error if needed
        end
        return nothing
    end
end

function precompute_thread_chunks!(buffers::SimulationBuffers{T}, n_particles::Int) where T<:Float64
    n_threads = Threads.nthreads()
    chunk_size = max(1, n_particles ÷ n_threads)
    
    # Allocate once
    buffers.thread_chunks = Vector{UnitRange{Int}}(undef, n_threads)
    
    # Calculate chunk ranges for each thread
    for i in 1:n_threads
        chunk_start = (i-1) * chunk_size + 1
        chunk_end = min(i * chunk_size, n_particles)
        buffers.thread_chunks[i] = chunk_start:chunk_end
    end
end