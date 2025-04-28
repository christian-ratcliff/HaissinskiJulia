# src/gpu_wakefield.jl
"""
    gpu_wakefield.jl - GPU implementation of wakefield effects

This file implements the wakefield calculations for particles
using CUDA acceleration. The implementation requires GPU for histogram,
FFT convolution, and interpolation.
"""

using CUDA
using CUDA.CUFFT

"""
    histogram_kernel!(z, bin_counts, bin_start, bin_step, nbins)

CUDA kernel for calculating histogram of particle positions.

# Args
- `z::CuDeviceVector{T}`: Particle longitudinal positions
- `bin_counts::CuDeviceVector{Int}`: Output bin counts
- `bin_start::T`: First bin edge
- `bin_step::T`: Bin width
- `nbins::Int`: Number of bins
"""
function histogram_kernel!(z, bin_counts, bin_start, bin_step, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(z)
        # Calculate bin index for this particle
        bin_idx = floor(Int, (z[idx] - bin_start) / bin_step) + 1
        if bin_idx >= 1 && bin_idx <= nbins
            CUDA.atomic_add!(pointer(bin_counts, bin_idx), 1)
        end
    end
    return nothing
end

"""
    calculate_histogram_gpu!(
        z::CuVector{T},
        gpu_buffers::GPUSimulationBuffers{T},
        gpu_config::GPUConfig
    ) where T<:Float64

Calculate histogram of particle positions on GPU.

# Args
- `z::CuVector{T}`: Particle longitudinal positions
- `gpu_buffers::GPUSimulationBuffers{T}`: GPU simulation buffers
- `gpu_config::GPUConfig`: GPU kernel configuration

# Returns
- `CuVector{Int}`: Bin counts on GPU
"""
function calculate_histogram_gpu!(
    z::CuVector{T},
    gpu_buffers::GPUSimulationBuffers{T},
    gpu_config::GPUConfig
) where T<:Float64
    nbins = length(gpu_buffers.bin_counts)
    n_particles = length(z)
    
    # Reset bin counts
    CUDA.fill!(gpu_buffers.bin_counts, 0)
    
    if n_particles == 0
        return gpu_buffers.bin_counts
    end
    
    # Get bin parameters
    bin_start = gpu_buffers.d_bin_edges[1]
    bin_step = (gpu_buffers.d_bin_edges[end] - bin_start) / nbins
    
    # Prepare kernel launch
    threads = gpu_config.threads_per_block
    blocks = min(
        cld(n_particles, threads), 
        gpu_config.max_blocks > 0 ? gpu_config.max_blocks : typemax(Int)
    )
    
    # Launch kernel
    if gpu_config.prefer_l1_cache
        @cuda prefer_l1=true blocks=blocks threads=threads histogram_kernel!(
            z, gpu_buffers.bin_counts, bin_start, bin_step, nbins
        )
    else
        @cuda blocks=blocks threads=threads histogram_kernel!(
            z, gpu_buffers.bin_counts, bin_start, bin_step, nbins
        )
    end
    
    return gpu_buffers.bin_counts
end

"""
    wake_function_kernel!(WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ, nbins)

CUDA kernel for calculating wake function values at bin centers.

# Args
- `WF_temp::CuDeviceVector{T}`: Output wake function values
- `bin_centers::CuDeviceVector{T}`: Bin centers
- `wake_factor::T`: Wake function factor
- `wake_sqrt::T`: Wake function sqrt parameter
- `inv_cτ::T`: Inverse of cτ parameter
- `nbins::Int`: Number of bins
"""
function wake_function_kernel!(WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nbins
        z = bin_centers[idx]
        # Wake function definition requires z <= 0 (particle behind the source)
        WF_temp[idx] = z > 0 ? 0.0 : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    return nothing
end

"""
    delta_kernel!(λ, bin_centers, sigma_z, delta_std, nbins)

CUDA kernel for calculating Gaussian smoothing kernel.

# Args
- `λ::CuDeviceVector{T}`: Output smoothing kernel values
- `bin_centers::CuDeviceVector{T}`: Bin centers
- `sigma_z::T`: Bunch length
- `delta_std::T`: Smoothing width
- `nbins::Int`: Number of bins
"""
function delta_kernel!(λ, bin_centers, sigma_z, delta_std, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nbins
        x = bin_centers[idx]
        σ = delta_std
        
        if σ <= 0
            # Handle edge case
            λ[idx] = abs(x) < 1e-10 ? 1.0 / (sqrt(2 * π) * 1e-10) : 0.0
        else
            # Normal Gaussian kernel
            σ_inv_sqrt2pi = 1 / (sqrt(2 * π) * σ)
            exp_factor = -0.5 / (σ * σ)
            λ[idx] = σ_inv_sqrt2pi * exp(x * x * exp_factor)
        end
    end
    return nothing
end

"""
    normalize_scale_kernel!(normalized_λ, bin_counts, n_particles_global, nbins)

CUDA kernel for normalizing histogram counts.

# Args
- `normalized_λ::CuDeviceVector{T}`: Output normalized values
- `bin_counts::CuDeviceVector{Int}`: Bin counts
- `n_particles_global::Int`: Total number of particles
- `nbins::Int`: Number of bins
"""
function normalize_scale_kernel!(normalized_λ, bin_counts, n_particles_global, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nbins
        normalized_λ[idx] = bin_counts[idx] / n_particles_global
    end
    return nothing
end

"""
    multiply_lambda_kernel!(λ, normalized_λ, nbins)

CUDA kernel for multiplying lambda and normalized lambda.

# Args
- `λ::CuDeviceVector{T}`: Lambda values (smoothing kernel)
- `normalized_λ::CuDeviceVector{T}`: Normalized histogram values
- `nbins::Int`: Number of bins
"""
function multiply_lambda_kernel!(λ, normalized_λ, nbins)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nbins
        λ[idx] = λ[idx] * normalized_λ[idx]
    end
    return nothing
end

"""
    copy_to_complex_kernel!(d_complex, d_real, n)

CUDA kernel for copying real values to complex buffer.

# Args
- `d_complex::CuDeviceVector{Complex{T}}`: Output complex buffer
- `d_real::CuDeviceVector{T}`: Input real values
- `n::Int`: Number of values to copy
"""
function copy_to_complex_kernel!(d_complex, d_real, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        d_complex[idx] = Complex{eltype(d_real)}(d_real[idx])
    end
    return nothing
end

"""
    complex_multiply_kernel!(result, a, b, n, scale)

CUDA kernel for element-wise complex multiplication.

# Args
- `result::CuDeviceVector{Complex{T}}`: Output result buffer
- `a::CuDeviceVector{Complex{T}}`: First input buffer
- `b::CuDeviceVector{Complex{T}}`: Second input buffer
- `n::Int`: Number of values
- `scale::T`: Scale factor
"""
function complex_multiply_kernel!(result, a, b, n, scale)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        result[idx] = a[idx] * b[idx] * scale
    end
    return nothing
end

"""
    extract_real_kernel!(d_real, d_complex, n)

CUDA kernel for extracting real parts from complex buffer.

# Args
- `d_real::CuDeviceVector{T}`: Output real buffer
- `d_complex::CuDeviceVector{Complex{T}}`: Input complex buffer
- `n::Int`: Number of values to extract
"""
function extract_real_kernel!(d_real, d_complex, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        d_real[idx] = real(d_complex[idx])
    end
    return nothing
end

"""
    interpolation_kernel!(result, positions, bin_centers, potential_values, nbins, n_particles)

CUDA kernel for linear interpolation of potential values.

# Args
- `result::CuDeviceVector{T}`: Output interpolated values
- `positions::CuDeviceVector{T}`: Particle positions for interpolation
- `bin_centers::CuDeviceVector{T}`: Bin centers (x-grid)
- `potential_values::CuDeviceVector{T}`: Potential values at bin centers
- `nbins::Int`: Number of bins
- `n_particles::Int`: Number of particles
"""
function interpolation_kernel!(result, positions, bin_centers, potential_values, nbins, n_particles)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_particles
        pos = positions[idx]
        
        # Find the bin index using binary search
        bin_idx = 1
        if pos <= bin_centers[1]
            bin_idx = 1
        elseif pos >= bin_centers[nbins]
            bin_idx = nbins - 1
        else
            # Binary search
            low = 1
            high = nbins
            while high - low > 1
                mid = (low + high) ÷ 2
                if pos < bin_centers[mid]
                    high = mid
                else
                    low = mid
                end
            end
            bin_idx = low
        end
        
        # Linear interpolation
        x0 = bin_centers[bin_idx]
        x1 = bin_centers[bin_idx + 1]
        y0 = potential_values[bin_idx]
        y1 = potential_values[bin_idx + 1]
        
        # Calculate weight
        w = (pos - x0) / (x1 - x0)
        
        # Interpolate
        result[idx] = y0 * (1.0 - w) + y1 * w
    end
    return nothing
end

"""
    apply_potential_kernel!(ΔE, potential, n_particles)

CUDA kernel for applying potential to particle energy deviations.

# Args
- `ΔE::CuDeviceVector{T}`: Particle energy deviations
- `potential::CuDeviceVector{T}`: Potential values
- `n_particles::Int`: Number of particles
"""
function apply_potential_kernel!(ΔE, potential, n_particles)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n_particles
        ΔE[idx] -= potential[idx]
    end
    return nothing
end

"""
    apply_wakefield_gpu!(
        gpu_data::GPUParticleData{T},
        gpu_buffers::GPUSimulationBuffers{T},
        wake_factor::T,
        wake_sqrt::T,
        cτ::T,
        current::T,
        sigma_z::T,
        gpu_config::GPUConfig,
        n_particles_global::Int,
        comm::Union{MPI.Comm, Nothing}=nothing,
        use_mpi::Bool=false
    ) where T<:Float64

Apply wakefield effects to particles on GPU.

# Args
- `gpu_data::GPUParticleData{T}`: Particle data on GPU
- `gpu_buffers::GPUSimulationBuffers{T}`: GPU simulation buffers
- `wake_factor::T`: Wake function factor
- `wake_sqrt::T`: Wake function sqrt parameter
- `cτ::T`: Wake function cτ parameter
- `current::T`: Scaled beam current
- `sigma_z::T`: Bunch length (global)
- `gpu_config::GPUConfig`: GPU kernel configuration
- `n_particles_global::Int`: Total number of particles (global)
- `comm::Union{MPI.Comm, Nothing}`: MPI communicator (optional)
- `use_mpi::Bool`: Flag to enable MPI
"""
function apply_wakefield_gpu!(
    gpu_data::GPUParticleData{T},
    gpu_buffers::GPUSimulationBuffers{T},
    wake_factor::T,
    wake_sqrt::T,
    cτ::T,
    current::T,
    sigma_z::T,
    gpu_config::GPUConfig,
    n_particles_global::Int,
    comm::Union{MPI.Comm, Nothing}=nothing,
    use_mpi::Bool=false
) where T<:Float64
    n_particles = length(gpu_data.z)
    if n_particles == 0 && use_mpi
        # In MPI mode, still need to participate in communication
    elseif n_particles == 0 && !use_mpi
        return nothing
    end
    
    # Common parameters
    inv_cτ = (cτ == 0) ? T(Inf) : 1 / cτ
    nbins = length(gpu_buffers.λ)
    
    # Prepare thread configuration for bin operations
    bin_threads = min(gpu_config.threads_per_block, nbins)
    bin_blocks = cld(nbins, bin_threads)
    
    # Prepare thread configuration for particle operations
    particle_threads = gpu_config.threads_per_block
    particle_blocks = min(
        cld(n_particles, particle_threads),
        gpu_config.max_blocks > 0 ? gpu_config.max_blocks : typemax(Int)
    )
    
    # Calculate histogram on GPU
    calculate_histogram_gpu!(gpu_data.z, gpu_buffers, gpu_config)
    
    if use_mpi
        # MPI communication for histogram
        if comm === nothing
            error("MPI mode selected, but MPI communicator is Nothing.")
        end
        
        # Copy bin counts back to CPU for MPI Allreduce
        host_bin_counts = Array(gpu_buffers.bin_counts)
        
        # Create buffer for MPI Allreduce
        global_bin_counts = zeros(Int, nbins)
        
        # Allreduce local histograms -> global histogram
        MPI.Allreduce!(host_bin_counts, global_bin_counts, MPI.SUM, comm)
        
        # Copy global counts back to GPU
        copyto!(gpu_buffers.bin_counts, global_bin_counts)
    end
    
    # Calculate wake function values at bin centers
    @cuda blocks=bin_blocks threads=bin_threads wake_function_kernel!(
        gpu_buffers.d_WF_temp, gpu_buffers.d_bin_centers, 
        wake_factor, wake_sqrt, inv_cτ, nbins
    )
    
    # Calculate smoothed global lambda using Gaussian kernel
    delta_std = sigma_z > 0 ? T(0.15) : T(1.0)
    
    @cuda blocks=bin_blocks threads=bin_threads delta_kernel!(
        gpu_buffers.λ, gpu_buffers.d_bin_centers, sigma_z, delta_std, nbins
    )
    
    # Normalize bin counts
    @cuda blocks=bin_blocks threads=bin_threads normalize_scale_kernel!(
        gpu_buffers.normalized_λ, gpu_buffers.bin_counts, n_particles_global, nbins
    )
    
    # Calculate smoothed density by multiplying lambda (kernel) and normalized counts
    @cuda blocks=bin_blocks threads=bin_threads multiply_lambda_kernel!(
        gpu_buffers.λ, gpu_buffers.normalized_λ, nbins
    )
    
    # Prepare for FFT convolution
    power_2_length = length(gpu_buffers.d_fft_buffer1)
    
    # Clear FFT buffers
    CUDA.fill!(gpu_buffers.d_fft_buffer1, Complex{T}(0))
    CUDA.fill!(gpu_buffers.d_fft_buffer2, Complex{T}(0))
    
    # Copy wake function and lambda data to complex buffers for FFT
    complex_threads = min(gpu_config.threads_per_block, nbins)
    complex_blocks = cld(nbins, complex_threads)
    
    @cuda blocks=complex_blocks threads=complex_threads copy_to_complex_kernel!(
        gpu_buffers.d_fft_buffer1, gpu_buffers.d_WF_temp, nbins
    )
    
    @cuda blocks=complex_blocks threads=complex_threads copy_to_complex_kernel!(
        gpu_buffers.d_fft_buffer2, gpu_buffers.λ, nbins
    )
    
    # Perform FFTs
    CUFFT.fft!(gpu_buffers.d_fft_buffer1)
    CUFFT.fft!(gpu_buffers.d_fft_buffer2)
    
    # Complex multiply in frequency domain
    fft_threads = min(gpu_config.threads_per_block, power_2_length)
    fft_blocks = cld(power_2_length, fft_threads)
    
    @cuda blocks=fft_blocks threads=fft_threads complex_multiply_kernel!(
        gpu_buffers.d_convol, gpu_buffers.d_fft_buffer1, gpu_buffers.d_fft_buffer2,
        power_2_length, current
    )
    
    # Perform inverse FFT
    CUFFT.ifft!(gpu_buffers.d_convol)
    
    # Extract real part for potential values
    @cuda blocks=complex_blocks threads=complex_threads extract_real_kernel!(
        gpu_buffers.d_potential_values, gpu_buffers.d_convol, nbins
    )
    
    # Interpolate potential values to particle positions
    @cuda blocks=particle_blocks threads=particle_threads interpolation_kernel!(
        gpu_buffers.potential, gpu_data.z, gpu_buffers.d_bin_centers,
        gpu_buffers.d_potential_values, nbins, n_particles
    )
    
    # Apply potential to particle energy deviations
    @cuda blocks=particle_blocks threads=particle_threads apply_potential_kernel!(
        gpu_data.ΔE, gpu_buffers.potential, n_particles
    )
    
    return nothing
end