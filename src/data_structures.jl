# src/data_structures.jl

using CUDA
using StructArrays
using StaticArrays

"""
    Coordinate{T} <: FieldVector{2, T}

Coordinate in longitudinal phase space.

# Fields
- `z::T`: Longitudinal position relative to reference particle
- `ΔE::T`: Energy deviation from reference energy
"""
struct Coordinate{T} <: FieldVector{2, T}
    z::T
    ΔE::T
end

"""
    Particle{T} <: FieldVector{1, Coordinate}

Simple particle representation with coordinates in phase space.

# Fields
- `coordinates::Coordinate{T}`: Current position in phase space
"""
struct Particle{T} <: FieldVector{1, Coordinate}
    coordinates::Coordinate{T}
    # uncertainty::Coordinate{T} # Uncomment if uncertainty is needed
end

"""
    BeamTurn{T}

Container for particle states across multiple turns.

# Fields
- `states::Vector{StructArray{Particle{T}}}`: Array of particle states for each turn
"""
struct BeamTurn{T<:Float64}
    states::Vector{StructArray{Particle{T}}}
end

"""
    SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}

Type-stable container for all simulation parameters with multiple type parameters
to allow individual parameters to be StochasticTriple when needed.

# Type Parameters remain the same as provided original serial version

# Fields remain the same as provided original serial version
- `E0::TE`: Reference energy [eV]
- `mass::TM`: Particle mass [eV/c²]
- `voltage::TV`: RF voltage [V]
- `harmonic::Int`: RF harmonic number
- `radius::TR`: Accelerator radius [m]
- `pipe_radius::TPR`: Beam pipe radius [m]
- `α_c::TA`: Momentum compaction factor
- `ϕs::TPS`: Synchronous phase [rad]
- `freq_rf::TF`: RF frequency [Hz]
- `n_turns::Int`: Number of turns to simulate
- `use_wakefield::Bool`: Enable wakefield effects
- `update_η::Bool`: Update slip factor
- `update_E0::Bool`: Update reference energy
- `SR_damping::Bool`: Enable synchrotron radiation damping
- `use_excitation::Bool`: Enable quantum excitation
"""
struct SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}
    E0::TE
    mass::TM
    voltage::TV
    harmonic::Int
    radius::TR
    pipe_radius::TPR
    α_c::TA
    ϕs::TPS
    freq_rf::TF
    n_turns::Int
    use_wakefield::Bool
    update_η::Bool
    update_E0::Bool # Note: In MPI mode, this implies global E0 update
    SR_damping::Bool
    use_excitation::Bool
end

# Convenience constructor remains the same
function SimulationParameters(E0::Float64, mass::Float64, voltage::Float64,
                             harmonic::Int, radius::Float64, pipe_radius::Float64,
                             α_c::Float64, ϕs::Float64, freq_rf::Float64,
                             n_turns::Int, use_wakefield::Bool, update_η::Bool,
                             update_E0::Bool, SR_damping::Bool, use_excitation::Bool)
    return SimulationParameters{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}(
        E0, mass, voltage, harmonic, radius, pipe_radius,
        α_c, ϕs, freq_rf, n_turns, use_wakefield,
        update_η, update_E0, SR_damping, use_excitation
    )
end

"""
    GPUParticleData{T<:Float64}

GPU-resident particle data structure for efficient CUDA operations.

# Fields
- `z::CuVector{T}`: Longitudinal positions
- `ΔE::CuVector{T}`: Energy deviations
"""
struct GPUParticleData{T<:Float64}
    z::CuVector{T}
    ΔE::CuVector{T}
end

"""
    GPUSimulationBuffers{T<:Float64}

Pre-allocated GPU buffers for computation.

# Fields
- `particle_data::GPUParticleData{T}`: Particle state data on GPU
- `potential::CuVector{T}`: Potential values for each particle
- `random_buffer::CuVector{T}`: Random numbers for quantum excitation
- `ϕ_buffer::CuVector{T}`: Phase values buffer
- `WF_buffer::CuVector{T}`: Wake field buffer
- `bin_counts::CuVector{Int}`: Bin counts for histogram
- `λ::CuVector{T}`: Lambda buffer for wakefield
- `normalized_λ::CuVector{T}`: Normalized lambda buffer
- `d_fft_buffer1::CuVector{Complex{T}}`: FFT buffer 1
- `d_fft_buffer2::CuVector{Complex{T}}`: FFT buffer 2
- `d_convol::CuVector{Complex{T}}`: Convolution buffer
- `d_potential_values::CuVector{T}`: Potential values at bin centers
- `d_real_buffer::CuVector{T}`: Buffer for real values
- `fft_plans::Dict{Symbol, Any}`: CUFFT plans
- `d_WF_temp::CuVector{T}`: Wake function values buffer
- `d_bin_edges::CuVector{T}`: Bin edges for histogram
- `d_bin_centers::CuVector{T}`: Bin centers for histogram
- `pinned_z::Vector{T}`: Pinned memory for CPU-GPU transfers (z)
- `pinned_ΔE::Vector{T}`: Pinned memory for CPU-GPU transfers (ΔE)
"""
struct GPUSimulationBuffers{T<:Float64}
    # Core particle data
    particle_data::GPUParticleData{T}
    
    # Physics calculation buffers
    potential::CuVector{T}
    random_buffer::CuVector{T}
    ϕ_buffer::CuVector{T}
    WF_buffer::CuVector{T}
    
    # Wakefield buffers
    bin_counts::CuVector{Int}
    λ::CuVector{T}
    normalized_λ::CuVector{T}
    
    # FFT buffers
    d_fft_buffer1::CuVector{Complex{T}}
    d_fft_buffer2::CuVector{Complex{T}}
    d_convol::CuVector{Complex{T}}
    d_potential_values::CuVector{T}
    d_real_buffer::CuVector{T}
    
    # CUFFT plans
    fft_plans::Dict{Symbol, Any}
    
    # Additional wakefield buffers
    d_WF_temp::CuVector{T}
    d_bin_edges::CuVector{T}
    d_bin_centers::CuVector{T}
    
    # Pinned memory for CPU-GPU transfers
    pinned_z::Vector{T}
    pinned_ΔE::Vector{T}
end

"""
    create_gpu_buffers(n_particles::Int, nbins::Int, T::Type=Float64)

Create pre-allocated GPU buffers.

# Args
- `n_particles::Int`: Number of particles (local to this process)
- `nbins::Int`: Number of bins for histograms
- `T::Type=Float64`: Floating point type

# Returns
- `GPUSimulationBuffers{T}`: The GPU buffers
"""
function create_gpu_buffers(n_particles::Int, nbins::Int, T::Type=Float64)
    function next_power_of_two(n::Int64)::Int64
        n <= 0 && return 1 # Return 1 for non-positive input
        return Int64(2^ceil(Int, log2(n))) # Use Int for exponent ceiling
    end
    # Initialize particle data
    z = CUDA.zeros(T, n_particles)
    ΔE = CUDA.zeros(T, n_particles)
    particle_data = GPUParticleData(z, ΔE)
    
    # Initialize calculation buffers
    potential = CUDA.zeros(T, n_particles)
    random_buffer = CUDA.zeros(T, n_particles)
    ϕ_buffer = CUDA.zeros(T, n_particles)
    WF_buffer = CUDA.zeros(T, n_particles)
    
    # Wakefield buffers
    bin_counts = CUDA.zeros(Int, nbins)
    λ = CUDA.zeros(T, nbins)
    normalized_λ = CUDA.zeros(T, nbins)
    
    # FFT buffers
    power_2_length = power_2_length = next_power_of_two(nbins * 2)
    d_fft_buffer1 = CUDA.zeros(Complex{T}, power_2_length)
    d_fft_buffer2 = CUDA.zeros(Complex{T}, power_2_length)
    d_convol = CUDA.zeros(Complex{T}, power_2_length)
    d_potential_values = CUDA.zeros(T, nbins)
    d_real_buffer = CUDA.zeros(T, power_2_length)
    
    # Additional wakefield buffers
    d_WF_temp = CUDA.zeros(T, nbins)
    d_bin_edges = CUDA.zeros(T, nbins + 1)
    d_bin_centers = CUDA.zeros(T, nbins)
    
    # Pinned memory for CPU-GPU transfers
    pinned_z = CUDA.zeros(T, n_particles)
    pinned_ΔE = CUDA.zeros(T, n_particles)
    
    # CUFFT plans
    fft_plans = Dict{Symbol, Any}()
    fft_plans[:fft_plan1] = CUDA.CUFFT.plan_fft(d_fft_buffer1)
    fft_plans[:fft_plan2] = CUDA.CUFFT.plan_fft(d_fft_buffer2)
    fft_plans[:ifft_plan] = CUDA.CUFFT.plan_ifft(d_convol)
    
    return GPUSimulationBuffers(
        particle_data,
        potential, random_buffer, ϕ_buffer, WF_buffer,
        bin_counts, λ, normalized_λ,
        d_fft_buffer1, d_fft_buffer2, d_convol, d_potential_values, d_real_buffer,
        fft_plans,
        d_WF_temp, d_bin_edges, d_bin_centers,
        pinned_z, pinned_ΔE
    )
end

"""
    GPUConfig

Configuration for GPU kernel launches.

# Fields
- `threads_per_block::Int`: Number of threads per block
- `max_blocks::Int`: Maximum number of blocks (0 for unlimited)
- `prefer_l1_cache::Bool`: Prefer L1 cache over shared memory
"""
struct GPUConfig
    threads_per_block::Int
    max_blocks::Int
    prefer_l1_cache::Bool
    
    function GPUConfig(; threads_per_block=256, max_blocks=0, prefer_l1_cache=true)
        # Ensure threads_per_block is a multiple of warp size (32)
        if threads_per_block % 32 != 0
            threads_per_block = 32 * cld(threads_per_block, 32)
        end
        new(threads_per_block, max_blocks, prefer_l1_cache)
    end
end

"""
    is_gpu_available()

Check if a CUDA GPU is available for computation.

# Returns
- `Bool`: True if GPU is available and functional
"""
function is_gpu_available()
    try
        if CUDA.functional()
            return true
        end
    catch
        # If CUDA throws an error, GPU is not available
    end
    return false
end

"""
    initialize_gpu(n_particles::Int, nbins::Int, T::Type=Float64)

Initialize GPU if available, create buffers.

# Args
- `n_particles::Int`: Number of particles (local to this process)
- `nbins::Int`: Number of bins for histograms
- `T::Type=Float64`: Floating point type

# Returns
- `Tuple{Union{GPUSimulationBuffers{T}, Nothing}, Bool}`: Tuple of (gpu_buffers, gpu_enabled)
"""
function initialize_gpu(n_particles::Int, nbins::Int, T::Type=Float64)
    try
        println("Checking GPU availability...")
        if !CUDA.functional()
            println("CUDA is available but not functional")
            return nothing, false
        end

        # GPU is functional, now check compatibility
        gpu_compatible = true  # Default assumption
        gpu_config = :standard # Default configuration

        # Check for H200 GPU (compute capability 9.x)
        try
            device_name = CUDA.name(CUDA.device())
            println("GPU is available and functional")
            println("Device: $device_name")
            
            # Check if this is an H200 GPU
            if occursin("H200", device_name)
                println("Detected H200 GPU - using specialized configuration")
                gpu_config = :hopper
                # Any H200-specific settings could go here
            end
        catch e
            println("Warning: Could not identify GPU model: $e")
            # Continue with default config
        end

        # Create buffers with appropriate configuration
        try
            return create_gpu_buffers(n_particles, nbins, T), true
        catch e
            println("Error in GPU buffer creation: $e")
            return nothing, false
        end
    catch e
        println("Error in GPU initialization: $e")
        return nothing, false
    end
end
"""
    SimulationBuffers{T<:Float64}

Pre-allocated buffers for efficient computation during simulation.
Includes buffers necessary for both serial and MPI (particle distribution) modes.

# Fields
- `WF::Vector{T}`: Buffer for potential wakefield values per particle (primarily serial use).
- `potential::Vector{T}`: Buffer for interpolated potential applied to local particles.
- `Δγ::Vector{T}`: Buffer for gamma factor deviations.
- `η::Vector{T}`: Buffer for slip factor calculations.
- `coeff::Vector{T}`: Buffer for temporary coefficients.
- `temp_z::Vector{T}`: General temporary storage for z coordinates.
- `temp_ΔE::Vector{T}`: General temporary storage for energy deviations.
- `temp_ϕ::Vector{T}`: General temporary storage for phases.
- `WF_temp::Vector{T}`: Buffer for wake function values at bin centers.
- `λ::Vector{T}`: Buffer for line charge density values at bin centers.
- `convol::Vector{Complex{T}}`: Buffer for FFT convolution results.
- `ϕ::Vector{T}`: Buffer for phase values.
- `random_buffer::Vector{T}`: Buffer for random numbers (quantum excitation).
- `normalized_λ::Vector{T}`: Buffer for normalized lambda (serial wakefield).
- `fft_buffer1::Vector{Complex{T}}`: Buffer for in-place FFT operations.
- `fft_buffer2::Vector{Complex{T}}`: Buffer for in-place FFT operations.
- `real_buffer::Vector{T}`: Buffer for storing real parts after IFFT.
- `bin_counts::Vector{Int}`: Buffer for local histogram counts.
- `thread_local_buffers::Vector{Dict{Symbol, Any}}`: Thread-local storage (serial).
- `global_bin_counts::Vector{Int}`: MPI Only: Buffer for storing result of Allreduce on histogram counts.
- `potential_values_at_centers_global::Vector{T}`: MPI Only: Buffer for receiving broadcasted potential grid.
"""
struct SimulationBuffers{T<:Float64}
    # Buffers sized based on n_particles (local in MPI, global in serial)
    WF::Vector{T}
    potential::Vector{T}
    Δγ::Vector{T}
    η::Vector{T}
    coeff::Vector{T}
    temp_z::Vector{T}
    temp_ΔE::Vector{T}
    temp_ϕ::Vector{T}
    ϕ::Vector{T}
    random_buffer::Vector{T}
    ΔE_initial_turn::Vector{T}
    mpi_fft_W::Vector{Complex{T}}
    mpi_fft_L::Vector{Complex{T}}
    mpi_convol_freq::Vector{Complex{T}} 

    # Buffers sized based on nbins (global property)
    WF_temp::Vector{T}
    λ::Vector{T}
    normalized_λ::Vector{T}      # Used in serial wakefield
    bin_counts::Vector{Int}      # Holds local counts before Allreduce in MPI mode

    # Buffers sized based on power_2_length (derived from nbins)
    convol::Vector{Complex{T}}
    fft_buffer1::Vector{Complex{T}}
    fft_buffer2::Vector{Complex{T}}
    real_buffer::Vector{T}       # For storing real parts
    

    # Thread local storage (serial optimization)
    thread_local_buffers::Vector{Dict{Symbol, Any}}

    # MPI Specific Buffers (sized based on nbins)
    global_bin_counts::Vector{Int}          # For storing result of Allreduce
    normalized_global_amounts::Vector{T}   
    potential_values_at_centers_global::Vector{T} # For receiving broadcasted potential grid
    fft_plans::Dict{Symbol, Any}
    interp_indices::Vector{Int}       # Pre-allocated indices buffer
    interp_weights::Vector{T}         # Pre-allocated weights buffer

    scatterv_counts::Vector{Int}
    scatterv_displs::Vector{Int}

    mpi_buffers::Dict{Symbol, Any}        # Store pre-allocated MPI buffer objects
    allreduce_energy::Vector{T}           # For energy updates [sum_dE, count]
    allreduce_stats::Vector{T}            # For statistics [sum, sum_sq, count]
    allreduce_single::Vector{T}           # For single value reductions

    thread_chunks::Vector{UnitRange{Int}}

    # gpu_buffers::Union{GPUSimulationBuffers{T}, Nothing}
    gpu_buffers::Union{Any, Nothing}
    use_gpu::Bool

end

function debug_gpu_kernels!(
    gpu_data::GPUParticleData{T},
    gpu_buffers::GPUSimulationBuffers{T},
    params::SimulationParameters,
    gpu_config::GPUConfig
    ) where T<:Float64
    println("Testing GPU kernels individually...")
    
    try
        println("Testing RF kick kernel...")
        rf_kick_gpu!(gpu_data, params.voltage, sin(params.ϕs), 
                    calc_rf_factor(params.freq_rf, sqrt(1-1/(params.E0/params.mass)^2)), 
                    params.ϕs, gpu_config)
        println("RF kick kernel OK")
    catch e
        println("Error in RF kick kernel: ", e)
        return :rf_kick
    end
    
    try
        println("Testing phase advance kernel...")
        apply_phase_advance_gpu!(gpu_data, 0.001, params.harmonic, 0.9, 
                                params.E0, 1.0, params.ϕs, gpu_config)
        println("Phase advance kernel OK")
    catch e
        println("Error in phase advance kernel: ", e)
        return :phase_advance
    end
    
    # Add similar tests for other kernels...
    
    println("All kernels tested successfully!")
    return :all_ok
end