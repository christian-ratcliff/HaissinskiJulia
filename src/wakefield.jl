# # """
# # wakefield.jl - Wakefield and collective effects

# # This file implements the wakefield calculations and collective effects
# # that couple particles within the beam, including:
# # - Wake function calculation
# # - Convolution with charge distribution
# # - Application of wakefield forces to particles
# # """

# # using LoopVectorization
# # using FFTW
# # using Interpolations
# # using FHist

# # """
# #     apply_wakefield_inplace!(
# #         particles::StructArray{Particle{T}},
# #         buffers::SimulationBuffers{T},
# #         wake_factor::T,
# #         wake_sqrt::T,
# #         cτ::T,
# #         current::T,
# #         σ_z::T,
# #         bin_edges
# #     ) where T<:Float64 -> Nothing

# # Apply wakefield effects to all particles.
# # """
# # function apply_wakefield_inplace!(
# #     particles::StructArray{Particle{T}},
# #     buffers::SimulationBuffers{T},
# #     wake_factor::T,
# #     wake_sqrt::T,
# #     cτ::T,
# #     current::T,
# #     σ_z::T,
# #     bin_edges
# #     ) where T<:Float64
    
# #     # Clear buffers
# #     fill!(buffers.λ, zero(T))
# #     fill!(buffers.WF_temp, zero(T))
# #     fill!(buffers.convol, zero(Complex{T}))
    
# #     # Get z positions
# #     n_particles = length(particles)
# #     inv_cτ::T = 1 / cτ
    
# #     # Calculate histogram
# #     bin_centers::Vector{T}, bin_amounts::Vector{T} = calculate_histogram(particles.coordinates.z, bin_edges)
# #     nbins::Int = length(bin_centers)
# #     power_2_length::Int = nbins * 2
    
    
# #     # Calculate line charge density using Gaussian smoothing
# #     delta_std::T = (15 * σ_z) / σ_z / 100
# #     @inbounds for i in eachindex(bin_centers)
# #         buffers.λ[i] = delta(bin_centers[i], delta_std)
# #     end
    
# #     # Calculate wake function for each bin
# #     @inbounds for i in eachindex(bin_centers)
# #         z = bin_centers[i]
# #         buffers.WF_temp[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
# #     end

# #     # Prepare arrays for convolution
# #     normalized_amounts = bin_amounts .* (1/n_particles)
# #     λ = buffers.λ[1:nbins]
# #     WF_temp = buffers.WF_temp[1:nbins]
# #     convol = buffers.convol[1:power_2_length]
    
# #     # Perform convolution and scale by current
# #     convol .= FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length) .* current

    
    
# #     # Interpolate results back to particle positions
# #     temp_z = range(minimum(particles.coordinates.z), maximum(particles.coordinates.z), length=length(convol))
# #     resize!(buffers.potential, length(particles.coordinates.z))
    
# #     # Create an interpolation function
# #     # itp = LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line())
# #     copyto!(buffers.temp_ΔE[1:length(convol)], real.(convol))
# #     itp = LinearInterpolation(temp_z, buffers.temp_ΔE[1:length(convol)], extrapolation_bc=Line())
    
# #     # Apply the interpolated potential to particles
# #     @inbounds for i in eachindex(particles.coordinates.z)
# #         z = particles.coordinates.z[i]
# #         potential_value = itp(z)
# #         particles.coordinates.ΔE[i] -= potential_value
# #         # buffers.WF[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
# #     end
    
# #     return nothing
# # end

# # """
# #     calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, cτ::T) where T<:Float64 -> T

# # Calculate the wake function for a given longitudinal position.
# # """
# # function calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64
# #     return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
# # end

# # """
# #     compute_line_charge_density(particles.coordinates.z::Vector{T}, σ_z::T, bin_edges) where T<:Float64 
# #                                -> Tuple{Vector{T}, Vector{T}}

# # Compute the line charge density distribution for a particle distribution.
# # """
# # function compute_line_charge_density(particles::StructArray{Particle{T}}, σ_z::T, bin_edges) where T<:Float64
# #     # Calculate histogram
# #     bin_centers, bin_amounts = calculate_histogram(particles.coordinates.z, bin_edges)
    
# #     # Normalize
# #     n_particles = length(particles.coordinates.z)
# #     normalized_amounts = bin_amounts .* (1/n_particles)
    
# #     # Apply Gaussian smoothing
# #     delta_std = (15 * σ_z) / σ_z / 100
# #     smoothed_density = Vector{T}(undef, length(bin_centers))
    
# #     for i in eachindex(bin_centers)
# #         smoothed_density[i] = delta(bin_centers[i], delta_std) * normalized_amounts[i]
# #     end
    
# #     return bin_centers, smoothed_density
# # end

# File: wakefield.jl (MPI Particle Distribution Version)


"""
wakefield.jl - Wakefield and collective effects (Particle Distribution Strategy)

Handles MPI communication for global histogram and potential broadcast.
"""

using LoopVectorization
using FFTW
using Interpolations
using FHist
using MPI

""" Calculate histogram for local data using global bin edges """
function calculate_local_histogram!(
    local_data::AbstractVector{T},
    bin_edges::AbstractRange,
    local_bin_counts::Vector{Int} # Output buffer for local counts
) where T<:Float64
    fill!(local_bin_counts, 0)
    nbins = length(local_bin_counts) # Should match length(bin_edges) - 1
    if length(bin_edges) != nbins + 1
         @error "Mismatch between bin_edges length ($(length(bin_edges))) and local_bin_counts length ($nbins)"
    end

    @inbounds for val in local_data
        # searchsortedfirst finds the index of the first edge >= val
        # The bin index is then edge_index - 1
        bin_idx = searchsortedfirst(bin_edges, val) - 1
        # Ensure the value falls within the defined bins
        if 1 <= bin_idx <= nbins
            local_bin_counts[bin_idx] += 1
        # else: particle is outside histogram range, ignore (or count under/overflow)
        end
    end
    return nothing
end

"""
    apply_wakefield_inplace!(...) - Particle Distribution Version

Gathers global histogram via Allreduce, Rank 0 computes FFTs, potential is
broadcasted, kick applied locally.
"""
function apply_wakefield_inplace!(
    particles::StructArray{Particle{T}}, # Holds N_local particles
    buffers::SimulationBuffers{T},      # Sized for N_local / nbins
    wake_factor::T,
    wake_sqrt::T,
    cτ::T,
    current::T,                       # Global wake current
    σ_z_global::T,                    # Global sigma_z for smoothing consistency
    bin_edges::AbstractRange{T}       # Global bin edges
    ) where T<:Float64

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    n_local = length(particles)
    inv_cτ::T = 1 / cτ
    # Get nbins (global property) from buffers
    nbins::Int = length(buffers.λ) # e.g., length of WF_temp, λ, global_bin_counts etc.
    if nbins <= 0; error("apply_wakefield_inplace: nbins is zero or negative."); end
    if length(bin_edges) != nbins + 1; error("apply_wakefield_inplace: bin_edges length mismatch."); end

    # Calculate bin_centers (all ranks do this)
    bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) ./ 2

    # --- Step 1: Calculate LOCAL histogram ---
    # Use the pre-allocated local 'bin_counts' buffer
    calculate_local_histogram!(particles.coordinates.z, bin_edges, buffers.bin_counts)

    # --- Step 2: Allreduce local histograms to get GLOBAL histogram ---
    # Sum local counts into the global_bin_counts buffer
    # Ensure global_bin_counts buffer exists and has the right size (nbins)
    @assert length(buffers.global_bin_counts) == nbins "global_bin_counts buffer size mismatch"
    MPI.Allreduce!(buffers.bin_counts, buffers.global_bin_counts, MPI.SUM, comm)
    # Now buffers.global_bin_counts holds the sum on all ranks

    # --- Step 3: Rank 0 prepares functions for convolution (using GLOBAL counts) ---
    local power_2_length::Int # Declare for scope
    if rank == 0
        # Get total number of particles from the reduction
        n_particles_global = sum(buffers.global_bin_counts)
        if n_particles_global == 0
             # Handle case with no particles - skip wakefield?
             # For now, proceed but expect zeros. Need robust handling.
             @warn "Global particle count is zero in wakefield calculation."
        end

        # Calculate global line charge density (λ) using GLOBAL histogram
        # Use buffers.λ and buffers.WF_temp (size nbins)
        fill!(@view(buffers.λ[1:nbins]), zero(T))
        fill!(@view(buffers.WF_temp[1:nbins]), zero(T))

        global_bin_amounts = T.(buffers.global_bin_counts) # Convert global counts to Float

        # Smoothing factor based on GLOBAL sigma_z
        delta_std::T = (15 * σ_z_global) / σ_z_global / 100 # = 0.15 if sigma_z_global > 0
        delta_std = σ_z_global > 0 ? T(0.15) : T(1.0) # Avoid NaN, use large sigma if global sigma is 0

        @inbounds for i in 1:nbins
            buffers.λ[i] = delta(bin_centers[i], delta_std) # Use delta smoothing function
        end

        # Calculate wake function (WF_temp)
        @inbounds for i in 1:nbins
            z = bin_centers[i]
            buffers.WF_temp[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
        end

        # Normalize global amounts for convolution input
        # Handle n_particles_global = 0 case
        inv_n_global = n_particles_global > 0 ? (1 / T(n_particles_global)) : zero(T)
        normalized_global_amounts = global_bin_amounts .* inv_n_global

        # Get FFT length from buffers
        power_2_length = length(buffers.fft_buffer1)

        # Prepare FFT inputs (using GLOBAL lambda * normalized_global_amounts)
        fft_W = buffers.fft_buffer1; fft_L = buffers.fft_buffer2
        fill!(fft_W, zero(Complex{T})); fill!(fft_L, zero(Complex{T}))
        @inbounds for i in 1:nbins
            fft_W[i] = Complex{T}(buffers.WF_temp[i])
            fft_L[i] = Complex{T}(buffers.λ[i] * normalized_global_amounts[i]) # Use global density here
        end

        # Perform FFTs
        plan_fft_W = plan_fft!(fft_W); plan_fft_L = plan_fft!(fft_L)
        plan_fft_W * fft_W; plan_fft_L * fft_L

    else # Non-root ranks only need power_2_length
        power_2_length = length(buffers.fft_buffer1)
    end # End rank == 0 block

    # --- Step 4: MPI Parallel Convolution (Scatter/Multiply/Gather) ---
    power_2_length = MPI.Bcast(power_2_length, 0, comm) # Ensure consistency
    if power_2_length % comm_size != 0
        base_chunk_size = power_2_length ÷ comm_size; remainder = power_2_length % comm_size
        counts = fill(base_chunk_size, comm_size); counts[1:remainder] .+= 1; chunk_size = counts[rank+1]
        if rank == 0; displs = vcat([0], cumsum(counts)[1:end-1]); else; displs = nothing; end
    else
        chunk_size = power_2_length ÷ comm_size; counts = fill(chunk_size, comm_size)
        if rank == 0; displs = collect(0:chunk_size:(power_2_length - chunk_size)); else; displs = nothing; end
    end
    local_fft_W = Vector{Complex{T}}(undef, chunk_size); local_fft_L = Vector{Complex{T}}(undef, chunk_size)
    local_convol_freq = Vector{Complex{T}}(undef, chunk_size)
    rbuf_W = MPI.Buffer(local_fft_W); rbuf_L = MPI.Buffer(local_fft_L)
    if rank == 0
        sbuf_W = MPI.VBuffer(buffers.fft_buffer1, counts, displs); sbuf_L = MPI.VBuffer(buffers.fft_buffer2, counts, displs)
        MPI.Scatterv!(sbuf_W, rbuf_W, 0, comm); MPI.Scatterv!(sbuf_L, rbuf_L, 0, comm)
    else
        MPI.Scatterv!(nothing, rbuf_W, 0, comm); MPI.Scatterv!(nothing, rbuf_L, 0, comm)
    end
    # Local multiply with GLOBAL current
    @inbounds @simd for i in 1:chunk_size; local_convol_freq[i] = local_fft_W[i] * local_fft_L[i] * current; end
    # Gather
    sbuf_convol = MPI.Buffer(local_convol_freq)
    if rank == 0
        @assert length(buffers.convol) >= power_2_length; rbuf_convol = MPI.VBuffer(buffers.convol, counts, displs); MPI.Gatherv!(sbuf_convol, rbuf_convol, 0, comm)
    else
        MPI.Gatherv!(sbuf_convol, nothing, 0, comm)
    end

    # --- Step 5: Rank 0 computes IFFT, Extracts Potential Grid ---
    local potential_values_at_centers_global::Vector{T} # Declare for scope
    if rank == 0
        # IFFT
        plan_ifft = plan_ifft!(buffers.convol); plan_ifft * buffers.convol

        # Extract real part into potential_values_at_centers_global buffer
        @assert length(buffers.potential_values_at_centers_global) == nbins
        potential_values_at_centers_global = buffers.potential_values_at_centers_global # Alias for clarity
        @inbounds for i in 1:nbins
            potential_values_at_centers_global[i] = real(buffers.convol[i])
        end
    else
        # Other ranks need the buffer to receive the broadcast
        potential_values_at_centers_global = buffers.potential_values_at_centers_global
    end

    # --- Step 6: Broadcast Potential Grid from Rank 0 ---
    MPI.Bcast!(potential_values_at_centers_global, 0, comm)
    # Now all ranks have the potential grid in buffers.potential_values_at_centers_global

    # --- Step 7: Local Interpolation and Application ---
    # All ranks create the interpolation function using the GLOBAL potential grid
    itp = LinearInterpolation(bin_centers, potential_values_at_centers_global, extrapolation_bc=Line())

    # Each rank applies the kick ONLY to its LOCAL particles
    @inbounds @simd for i in 1:n_local
        potential_value = itp(particles.coordinates.z[i])
        particles.coordinates.ΔE[i] -= potential_value
    end

    return nothing
end

# calculate_wake_function remains the same
@inline function calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64
    return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
end