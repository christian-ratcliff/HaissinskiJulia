# src/wakefield.jl
"""
wakefield.jl - Wakefield and collective effects simulation

Implements wakefield calculation supporting both serial execution and
MPI-based particle distribution parallelization.
- Serial: Local histogram, FFT, convolution, interpolation.
- MPI: Local histogram, Allreduce for global histogram, Rank 0 computes
       FFT/convolution using Scatterv/Gatherv, Broadcasts potential grid,
       local interpolation applies kick.
"""

using LoopVectorization
using FFTW
using Interpolations
using FHist
using MPI # Needed for MPI operations
using FLoops

# Local helper functions (used by both modes potentially)

function zero_alloc_interpolation!(
    result::Vector{T},                # Output buffer (reuse buffers.potential)  
    particle_positions::Vector{T},    # Current particle positions
    bin_centers::AbstractRange{T},           # X-grid
    potential_values::Vector{T},      # Y-values (current potential)
    indices_buffer::Vector{Int},      # Reuse buffers.interp_indices
    weights_buffer::Vector{T},         # Reuse buffers.interp_weights
    buffers::SimulationBuffers{T}
) where T<:Float64
    
    nbins = length(bin_centers)
    if nbins <= 1
        fill!(result, nbins == 1 ? potential_values[1] : zero(T))
        return nothing
    end

    # Pre-compute range parameters
    bin_start = bin_centers[1]
    bin_end = bin_centers[end]
    bin_step = (bin_end - bin_start) / (nbins - 1)
    inv_step = 1.0 / bin_step


    # # Vectorize the calculation with @turbo
    # Threads.@threads for tid in 1:Threads.nthreads()
    #     chunk_range = buffers.thread_chunks[tid]
    #     @turbo for i in chunk_range
    #         pos = particle_positions[i]
            
    #         # Calculate normalized position
    #         normalized_pos = (clamp(pos, bin_start, bin_end) - bin_start) * inv_step
            
    #         # Calculate base index (floor)
    #         base_idx = floor(Int, normalized_pos) + 1
            
    #         # Store clamped index (1 to nbins-1)
    #         indices_buffer[i] = clamp(base_idx, 1, nbins-1)
            
    #         # Calculate weight (fractional part)
    #         weights_buffer[i] = normalized_pos - floor(normalized_pos)
    #     end
    # end
    @turbo for i in 1:length(particle_positions)
        pos = particle_positions[i]
        
        # Calculate normalized position
        normalized_pos = (clamp(pos, bin_start, bin_end) - bin_start) * inv_step
        
        # Calculate base index (floor)
        base_idx = floor(Int, normalized_pos) + 1
        
        # Store clamped index (1 to nbins-1)
        indices_buffer[i] = clamp(base_idx, 1, nbins-1)
        
        # Calculate weight (fractional part)
        weights_buffer[i] = normalized_pos - floor(normalized_pos)
    end

    # Apply interpolation (separate loop for better cache behavior)
    # Threads.@threads for tid in 1:Threads.nthreads()
        #     chunk_range = buffers.thread_chunks[tid]
        #     @turbo for i in chunk_range
    #         idx = indices_buffer[i]
    #         w = weights_buffer[i]
            
    #         # Linear interpolation
    #         v1 = potential_values[idx]
    #         v2 = potential_values[idx+1]
    #         result[i] = v1 * (1.0 - w) + v2 * w
    #     end
    # end

    @inbounds for i in 1:length(weights_buffer)
        idx = indices_buffer[i]
        w = weights_buffer[i]
        
        # Linear interpolation
        v1 = potential_values[idx]
        v2 = potential_values[idx+1]
        result[i] = v1 * (1.0 - w) + v2 * w
    end

    return nothing
end


"""
    calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64 -> T

Calculate the wake function value for a given longitudinal position z.
Assumes z=0 is the head of the bunch, positive z is upstream (arriving earlier).
Wake is typically zero for z > 0 (causality).
"""
@inline function calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64
    # Wake function definition requires z <= 0 (particle behind the source)
    return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
end


"""
    apply_wakefield_inplace!(...)

Apply wakefield effects to particles. Selects serial or MPI implementation based on flag.

Args:
    particles (StructArray): Particle data (local partition in MPI mode).
    buffers (SimulationBuffers): Pre-allocated buffers.
    params (SimulationParameters): Simulation parameters.
    current (T): Scaled beam current for wakefield amplitude.
    sigma_z (T): Bunch length (global value if use_mpi=true, local otherwise).
                 Used for smoothing parameter calculation.
    bin_edges (AbstractRange): Globally defined bin edges for histogram.
    comm (MPI.Comm or Nothing): MPI communicator (Nothing for serial).
    use_mpi (Bool): Flag to select execution mode.
"""
function apply_wakefield_inplace!(
    particles::StructArray{Particle{T}},
    buffers::SimulationBuffers{T},
    wake_factor::T,
    wake_sqrt::T,
    cτ::T,
    current::T,
    sigma_z::T, # This should be sigma_z_global in MPI mode
    bin_edges::AbstractRange{T},
    bin_centers::AbstractRange{T},
    comm::Union{MPI.Comm, Nothing}, # MPI communicator or nothing
    use_mpi::Bool # Flag for mode selection
    ) where T<:Float64

    n_particles = length(particles) # Local count in MPI, total in serial
    if n_particles == 0 && use_mpi
    elseif n_particles == 0 && !use_mpi
         return nothing # No particles to apply wakefield to in serial mode
    end

    # Common parameters
    inv_cτ = (cτ == 0) ? T(Inf) : 1 / cτ # Avoid division by zero
    nbins = length(buffers.λ) # Number of bins from buffer size
    if nbins <= 0; error("apply_wakefield_inplace: nbins is zero or negative."); end
    if length(bin_edges) != nbins + 1; error("apply_wakefield_inplace: bin_edges length mismatch."); end
    power_2_length = length(buffers.fft_buffer1) # FFT length from buffer size
    if power_2_length < 2*nbins error("FFT buffer length too small for convolution"); end

    # --- MPI Mode ---
    if use_mpi
        if comm === nothing
            error("MPI mode selected, but MPI communicator is Nothing.")
        end
        rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)

        # Step 1: Calculate LOCAL histogram (using local particle data)
        # Use local bin_counts buffer
        calculate_local_histogram!(particles.coordinates.z, bin_edges, buffers.bin_counts)

        # Step 2: Allreduce local histograms -> GLOBAL histogram
        # Result stored in global_bin_counts buffer on all ranks
        @assert length(buffers.global_bin_counts) == nbins "global_bin_counts buffer size mismatch"
        MPI.Allreduce!(buffers.bin_counts, buffers.global_bin_counts, MPI.SUM, comm)
        # buffers.global_bin_counts now holds the global histogram on all ranks

        # Step 3: Rank 0 prepares functions for convolution (using GLOBAL counts)
        local potential_values_at_centers_global::Vector{T} # Define scope for broadcast target

        if rank == 0
            n_particles_global = sum(buffers.global_bin_counts)

            # --- Rank 0: Calculate wake function and smoothed global lambda ---
            fill!(buffers.WF_temp, zero(T)) # Wake function at bin centers
            fill!(buffers.λ, zero(T))       # Smoothed global lambda at bin centers
            fill!(buffers.random_buffer, zero(T))
            # Calculate wake function values at bin centers
            @inbounds for i in 1:nbins
                buffers.WF_temp[i] = calculate_wake_function(bin_centers[i], wake_factor, wake_sqrt, inv_cτ)
            end

            # @floop ThreadedEx() for i in 1:nbins
            #     @inbounds buffers.WF_temp[i] = calculate_wake_function(bin_centers[i], wake_factor, wake_sqrt, inv_cτ)
            # end

            # Calculate smoothed global lambda using Gaussian kernel
            # Smoothing width depends on GLOBAL sigma_z
            delta_std = 0.15 # Consistent smoothing parameter = 0.15 if sigma_z > 0

            # Direct smoothing using the delta function and global counts
            inv_n_global = 1 / n_particles_global
            
            @turbo for i in 1:nbins
                buffers.normalized_global_amounts[i] = buffers.global_bin_counts[i] * inv_n_global
            end

            

            # lambda_kernel = Vector{T}(undef, nbins)
            lambda_kernel = view(buffers.random_buffer, 1:nbins)


            @inbounds @simd for i in 1:nbins
                lambda_kernel[i] = delta(bin_centers[i], delta_std) # Use the delta function
            end
            
            # @floop ThreadedEx() for i in 1:nbins
            #     @inbounds lambda_kernel[i] = delta(bin_centers[i], delta_std)
            # end
            

            copyto!(buffers.λ, lambda_kernel)

            # --- Rank 0: Prepare for FFT Convolution ---
            fft_W = buffers.fft_buffer1; fft_L = buffers.fft_buffer2
            fill!(fft_W, zero(Complex{T})); fill!(fft_L, zero(Complex{T}))

            @inbounds for i in 1:nbins
                 # Wake function values
                fft_W[i] = Complex{T}(buffers.WF_temp[i])
                 # Smoothed density = kernel * normalized counts
                fft_L[i] = Complex{T}(buffers.λ[i] * buffers.normalized_global_amounts[i])
            end

            # Perform FFTs (in-place)
            buffers.fft_plans[:fft_plan1] * fft_W
            buffers.fft_plans[:fft_plan2] * fft_L   
            # fft_W and fft_L now contain the FFTs
        end # End rank == 0 FFT preparation block

        # --- Step 4: MPI Parallel Convolution using Scatterv/Gatherv ---
        # Calculate Scatterv/Gatherv parameters
        base_chunk_size = power_2_length ÷ comm_size
        remainder = power_2_length % comm_size
        counts = buffers.scatterv_counts
        displs = buffers.scatterv_displs
        current_displacement = 0
        for r in 0:(comm_size-1)
            local_chunk = r < remainder ? base_chunk_size + 1 : base_chunk_size
            counts[r+1] = local_chunk 
            displs[r+1] = current_displacement
            current_displacement += local_chunk
        end
        chunk_size = counts[rank+1] # This rank's chunk size

        # Create views for local chunks
        local_fft_W = view(buffers.mpi_fft_W, 1:chunk_size)
        local_fft_L = view(buffers.mpi_fft_L, 1:chunk_size)
        local_convol_freq = view(buffers.mpi_convol_freq, 1:chunk_size)

        # --- MPI Buffer Management ---
        # Get references to the buffers (safely)
        rbuf_W = buffers.mpi_buffers[:rbuf_W]
        rbuf_L = buffers.mpi_buffers[:rbuf_L]
        
        # Send buffers are only on rank 0
        sbuf_W = nothing
        sbuf_L = nothing
        if rank == 0
            sbuf_W = buffers.mpi_buffers[:sbuf_W]
            sbuf_L = buffers.mpi_buffers[:sbuf_L]
        end

        # Scatter the FFT data
        MPI.Scatterv!(sbuf_W, rbuf_W, 0, comm)
        MPI.Scatterv!(sbuf_L, rbuf_L, 0, comm)

        # Perform local element-wise multiplication in frequency domain
        # Apply the global 'current' scaling factor here
        @inbounds @simd for i in 1:chunk_size
            local_convol_freq[i] = local_fft_W[i] * local_fft_L[i] * current
        end

        # Gather results back to Rank 0 into buffers.convol
        sbuf_convol = buffers.mpi_buffers[:sbuf_convol]
        rbuf_convol = nothing
        
        if rank == 0
            @assert length(buffers.convol) >= power_2_length "Convolution buffer too short"
            rbuf_convol = buffers.mpi_buffers[:rbuf_convol]
        end
        
        MPI.Gatherv!(sbuf_convol, rbuf_convol, 0, comm)
        # Now buffers.convol on Rank 0 holds the full frequency-domain result

        # --- Step 5: Rank 0 computes IFFT, Extracts Potential Grid ---
        if rank == 0
            # Only perform IFFT if there were particles
            if sum(buffers.global_bin_counts) > 0
                # In-place IFFT on buffers.convol
                buffers.fft_plans[:ifft_plan] * buffers.convol
            else
                 # If no particles, convol buffer should already be zero from Gather
                 fill!(buffers.convol, zero(Complex{T}))
            end

            # Extract real part into the designated global potential buffer
            potential_values_at_centers_global = buffers.potential_values_at_centers_global # Alias
            @inbounds for i in 1:nbins
                 # Take real part of the IFFT result
                 # The result of linear convolution should be real
                 potential_values_at_centers_global[i] = real(buffers.convol[i])
            end
        else
            # Other ranks need the buffer alias to receive the broadcast
            potential_values_at_centers_global = buffers.potential_values_at_centers_global
        end

        # --- Step 6: Broadcast Potential Grid from Rank 0 ---
        # potential_values_at_centers_global acts as both send (rank 0) and receive buffer
        MPI.Bcast!(potential_values_at_centers_global, 0, comm)
        # Now all ranks have the potential grid in buffers.potential_values_at_centers_global

        # --- Step 7: Local Interpolation and Application ---
        zero_alloc_interpolation!(
            buffers.potential,                          # Reuse existing buffer
            particles.coordinates.z,                    # Current positions
            bin_centers,                                # Grid (constant during a turn)
            buffers.potential_values_at_centers_global, # Current potential values
            buffers.interp_indices,                     # Pre-allocated indices buffer
            buffers.interp_weights,                      # Pre-allocated weights buffer
            buffers
        )

        # Apply potential directly from buffer
        @turbo for i in 1:n_particles
            particles.coordinates.ΔE[i] -= buffers.potential[i]
        end

    # --- Serial Mode ---
    else # if not use_mpi
        # Clear potentially unused MPI buffers
        fill!(buffers.global_bin_counts, 0)
        fill!(buffers.potential_values_at_centers_global, 0.0)

        # Step 1: Calculate Histogram (using global particle data)
        # Uses calculate_histogram helper which calls FHist internally
        # Store results directly in buffers.λ (centers) and buffers.bin_counts
        # Need centers calculation separate from histogram counts storage
        _centers, _counts = calculate_histogram(particles.coordinates.z, bin_edges)
        copyto!(buffers.bin_counts, _counts) # Store counts in the buffer

        # Step 2: Calculate Line Charge Density (lambda) with smoothing
        # Uses serial logic: smooth normalized amounts
        n_total_particles = sum(buffers.bin_counts) # Should equal n_particles in serial
        if n_total_particles == 0; return nothing; end # No wake if no particles

        normalized_amounts = buffers.bin_counts .* (1 / T(n_total_particles))
            

        # Smoothing factor depends on LOCAL sigma_z (which is global in serial)
        delta_std = sigma_z > 0 ? T(0.15) : T(1.0)

        # Store smoothed lambda in buffers.λ
        fill!(buffers.λ, zero(T))
        @inbounds for i in 1:nbins
            buffers.λ[i] = delta(bin_centers[i], delta_std) # Store kernel in lambda buffer
        end
        # Store normalized amounts in the designated buffer if needed elsewhere, or use directly
        copyto!(buffers.normalized_λ, normalized_amounts) # Store normalized amounts

        # Step 3: Calculate Wake Function
        fill!(buffers.WF_temp, zero(T))
        @inbounds for i in 1:nbins
            buffers.WF_temp[i] = calculate_wake_function(bin_centers[i], wake_factor, wake_sqrt, inv_cτ)
        end

        # Step 4: Perform Convolution
        # Convolve WF_temp with (lambda_kernel * normalized_amounts)
        # Use in-place convolution with buffers
        # Prepare g = lambda_kernel * normalized_amounts
        temp_g = Vector{T}(undef, nbins)
        @inbounds for i in 1:nbins
            temp_g[i] = buffers.λ[i] * buffers.normalized_λ[i]
        end

        # Perform convolution: convol = ifft(fft(WF_temp) * fft(temp_g)) * current
        # Using in_place_convolution helper
        in_place_convolution!(
            buffers.convol,           # Result buffer
            buffers.WF_temp,          # f = wake function
            temp_g,                   # g = smoothed density
            power_2_length,
            buffers.fft_buffer1,      # Workspace 1
            buffers.fft_buffer2,       # Workspace 2
            buffers
        )
        # Scale result by current
        # buffers.convol .*= current
        @turbo for i in 1:length(buffers.convol)
            buffers.convol[i] *= current
        end

        # Step 5: Interpolate results back to particle positions
        # Extract real part of potential from convolution result
        potential_at_centers = buffers.potential_values_at_centers_global # Reuse buffer
        @inbounds for i in 1:nbins
            potential_at_centers[i] = real(buffers.convol[i])
        end

        zero_alloc_interpolation!(
            buffers.potential,                          # Reuse existing buffer
            particles.coordinates.z,                    # Current positions
            bin_centers,                                # Grid (constant during a turn)
            buffers.potential_values_at_centers_global, # Current potential values
            buffers.interp_indices,                     # Pre-allocated indices buffer
            buffers.interp_weights,                      # Pre-allocated weights buffer
            buffers
        )

        # Apply potential directly from buffer
        @turbo for i in 1:n_particles
            particles.coordinates.ΔE[i] -= buffers.potential[i]
        end

    end # End of if use_mpi / else block

    return nothing
end