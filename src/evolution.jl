# src/evolution.jl
"""
evolution.jl - Particle evolution and beam dynamics simulation

Implements the core longitudinal evolution algorithm supporting both serial
and MPI (particle distribution) modes. Handles multi-turn tracking including
RF cavity, synchrotron radiation, quantum excitation, and collective effects (wakefield).
MPI mode includes communication for global sums/means and wakefield calculation.
"""

using StructArrays
using StaticArrays
using Statistics
using Random
using Distributions
using ProgressMeter
using LoopVectorization
using MPI # Needed for MPI operations
using LinearAlgebra 



# --- Particle Generation (Common function, used differently in serial/MPI) ---
"""
    generate_particles(...)

Generate initial particle distribution based on specified means and standard deviations.
Handles multivariate normal distribution generation and edge cases without using goto.

Args:
    μ_z, μ_E (T): Mean longitudinal position and energy deviation.
    σ_z, σ_E (T): Standard deviations of position and energy.
    num_particles (Int): Number of particles to generate.
    energy (T): Reference energy E0.
    mass (T): Particle mass.
    ϕs (T): Synchronous phase.
    freq_rf (T): RF frequency.

Returns:
    Tuple{StructArray{Particle{T}}, T, T, T}: Generated particles, actual σ_E, actual σ_z, initial energy.
"""
function generate_particles(
    μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int,
    energy::T, mass::T, ϕs::T, freq_rf::T) where T<:Float64

    # Ensure num_particles is non-negative
    if num_particles <= 0
         # Return an empty StructArray of the correct type and zero spreads
         empty_coords = StructArray(Coordinate.(Vector{T}(), Vector{T}()))
         empty_particles = StructArray{Particle{T}}((coordinates=empty_coords,))
         return empty_particles, T(0), T(0), energy
    end

    z_vals = Vector{T}(undef, num_particles)
    ΔE_vals = Vector{T}(undef, num_particles)
    actual_σ_E::T = σ_E # Default to input sigma
    actual_σ_z::T = σ_z # Default to input sigma

    # Need at least 2 samples for covariance estimation
    if num_particles < 2
        # Generate independent samples if only 1 particle
        rng = Random.default_rng() # Use default RNG
        z_vals[1] = rand(rng, Normal(μ_z, σ_z))
        ΔE_vals[1] = rand(rng, Normal(μ_E, σ_E))
        # Cannot calculate std dev, return input values
    else
        # Sample for covariance matrix estimation
        initial_sample_size::Int = min(num_particles, 10_000)
        rng = Random.default_rng()
        z_samples = rand(rng, Normal(μ_z, σ_z), initial_sample_size)
        E_samples = rand(rng, Normal(μ_E, σ_E), initial_sample_size)

        # Compute covariance matrix, handle potential errors
        local Σ # Ensure Σ is defined in this scope
        try
            cov_zz = cov(z_samples) # Variance
            cov_EE = cov(E_samples) # Variance
            cov_zE = cov(z_samples, E_samples) # Covariance
            # Ensure variances are non-negative
            Σ = Symmetric([max(T(1e-18), cov_zz)     cov_zE;
                               cov_zE               max(T(1e-18), cov_EE)])
        catch e
            @warn "Covariance calculation failed: $e. Using diagonal matrix with input sigmas."
            # Fallback to diagonal matrix if covariance fails
            Σ = Symmetric(diagm([max(T(1e-18), σ_z^2), max(T(1e-18), σ_E^2)]))
        end

        # Ensure matrix is positive semi-definite for MvNormal
        min_eig = 0.0
        try min_eig = eigmin(Σ) catch; min_eig = -Inf; end
        # Add small diagonal term if not positive semi-definite
        if min_eig < 1e-12
             Σ = Σ + max(0, (1e-12 - min_eig)) * I # Add to diagonal
        end

        # Create multivariate normal distribution
        μ_vec = SVector{2,T}(μ_z, μ_E) # Mean vector
        dist_total = nothing         # Initialize to nothing
        mvnormal_success = false     # Flag to track if MvNormal was created
        try
             dist_total = MvNormal(μ_vec, Σ)
             mvnormal_success = true # Creation succeeded
        catch e
             @error "MvNormal creation failed with mean=$μ_vec, cov=\n$Σ\nError: $e"
             # Fallback to independent normals if MvNormal fails catastrophically
             @warn "Falling back to independent Normal distributions."
             rng_fallback = Random.default_rng()
             rand!(rng_fallback, Normal(μ_z, σ_z), z_vals)
             rand!(rng_fallback, Normal(μ_E, σ_E), ΔE_vals)
             # mvnormal_success remains false
        end

        # Generate correlated samples ONLY if MvNormal succeeded
        if mvnormal_success && dist_total !== nothing
             # Generate samples using the created distribution
             samples = rand(rng, dist_total, num_particles)  # Returns 2 × num_particles matrix
             copyto!(z_vals, 1, view(samples, 1, :), 1, num_particles) # Use view for potential efficiency
             copyto!(ΔE_vals, 1, view(samples, 2, :), 1, num_particles)
        # else: z_vals and ΔE_vals already contain the fallback values from the catch block
        end

        # Calculate actual spread from generated samples (only if > 1 particle)
        actual_σ_z = compute_std(z_vals) # Use local std calculation
        actual_σ_E = compute_std(ΔE_vals) # Use local std calculation
    end

    # Create the StructArray of Particles
    coords = StructArray(Coordinate.(z_vals, ΔE_vals))
    particles = StructArray{Particle{T}}((coordinates=coords,))

    # Return particles and the actual standard deviations of the generated distribution
    return particles, actual_σ_E, actual_σ_z, energy
end


# --- Helper Functions for Evolution (Local Operations) ---

"""
    apply_phase_advance!(...)

Apply phase advancement using constant slip factor η0. Operates locally.
"""
function apply_phase_advance!(
    particles::StructArray{Particle{T}}, # Local particles
    η0, harmonic, β0, E0, rf_factor, ϕs
    ) where T<:Float64
    n_local = length(particles)
    if n_local == 0 return nothing end

    # Avoid division by zero or non-physical values
    if β0 <= 0 || E0 == 0 || rf_factor == 0
        @warn "Non-physical parameters in apply_phase_advance!: β0=$β0, E0=$E0, rf_factor=$rf_factor. Skipping advance."
        return nothing
    end

    # Calculate coefficient for phase advance
    coeff = (2π * harmonic * η0 / (β0 * β0 * E0))

    @turbo for i in 1:n_local
        # Convert z to phase, apply advance, convert back to z
        # Using inline versions of z_to_ϕ and ϕ_to_z for performance
        ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs) # z_to_ϕ inline
        ϕ_i += coeff * particles.coordinates.ΔE[i]          # Apply energy-dependent advance
        particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor # ϕ_to_z inline
    end
    return nothing
end

"""
    safe_update_energy!(...)

Subtract a value from particle energy deviations in-place. Operates locally.
(Equivalent to subtract_mean_inplace! from utils).
"""
function safe_update_energy!(particles::StructArray{Particle{T}}, value_to_subtract) where T<:Float64
     n_local = length(particles)
     if n_local == 0 return nothing end
     # Directly subtract the value
     @turbo for i in 1:n_local
         particles.coordinates.ΔE[i] -= value_to_subtract
     end
     return nothing
end


# --- Main Evolution Function (Serial/MPI Capable) ---

# Add GPU support to longitudinal_evolve! in evolution.jl

"""
    longitudinal_evolve!(
        particles::StructArray{Particle{T}},
        params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
        buffers::SimulationBuffers{T},
        comm::Union{MPI.Comm, Nothing},
        use_mpi::Bool,
        use_gpu::Bool = buffers.use_gpu,
        gpu_config::Union{GPUConfig,Nothing} = nothing
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}

Evolve particles through the accelerator for multiple turns.
Supports serial, MPI-only, and CUDA+MPI execution modes.

Args:
    particles (StructArray): Particle data (local partition in MPI mode).
    params (SimulationParameters): Simulation parameters.
    buffers (SimulationBuffers): Pre-allocated buffers.
    comm (MPI.Comm or Nothing): MPI communicator (required if use_mpi=true).
    use_mpi (Bool): Flag to enable MPI mode.
    use_gpu (Bool): Flag to enable GPU acceleration (default: buffers.use_gpu).
    gpu_config (GPUConfig or Nothing): GPU configuration (default: None).

Returns:
    Tuple{T, T, TE}: Final energy spread (σ_E), bunch length (σ_z), reference energy (E0).
                    Spreads are global in MPI mode.
"""
function longitudinal_evolve!(
    particles::StructArray{Particle{T}},
    params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF},
    buffers::SimulationBuffers{T},
    comm::Union{MPI.Comm, Nothing},
    use_mpi::Bool,
    use_gpu::Bool = buffers.use_gpu,
    gpu_config::Union{GPUConfig,Nothing} = nothing
    ) where {T<:Float64, TE, TM, TV, TR, TPR, TA, TPS, TF}

    # Check if GPU is requested but not available
    if use_gpu
        if !@isdefined(GPUConfig) || !@isdefined(GPUSimulationBuffers) || buffers.gpu_buffers === nothing
            @warn "GPU requested but GPU support not available or buffers not initialized. Falling back to CPU."
            use_gpu = false
        end
    end
    
    # Set default GPU config if not provided
    if use_gpu && gpu_config === nothing
        gpu_config = GPUConfig()
    end
    
    # --- MPI Setup ---
    rank = 0
    comm_size = 1
    if use_mpi
        if comm === nothing
            error("MPI mode requires a valid MPI communicator.")
        end
        rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)
    end

    # --- Extract Parameters ---
    # E0 is mutable within the function, potentially updated globally
    local E0::TE = params.E0 # Initialize with parameter value
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
    update_E0_flag = params.update_E0 
    SR_damping = params.SR_damping
    use_excitation = params.use_excitation

    # --- Local Particle Count ---
    n_local = length(particles) # Number of particles handled by this rank/process

    # --- Initial Spreads (Global if MPI, Local if Serial) ---
    local σ_E0_initial::T
    local σ_z0_initial::T
    
    if use_gpu
        # Transfer particles to GPU
        transfer_particles_to_gpu!(
            buffers.gpu_buffers.particle_data, 
            particles,
            buffers.gpu_buffers.pinned_z,
            buffers.gpu_buffers.pinned_ΔE
        )
        
        # Calculate initial spreads using GPU data
        if use_mpi
            σ_E0_initial = compute_global_std_gpu(buffers.gpu_buffers.particle_data.ΔE, comm, true)
            σ_z0_initial = compute_global_std_gpu(buffers.gpu_buffers.particle_data.z, comm, true)
        else
            σ_E0_initial = compute_global_std_gpu(buffers.gpu_buffers.particle_data.ΔE, comm, false)
            σ_z0_initial = compute_global_std_gpu(buffers.gpu_buffers.particle_data.z, comm, false)
        end
    else
        if use_mpi
            σ_E0_initial = compute_global_std(particles.coordinates.ΔE, comm, buffers)
            σ_z0_initial = compute_global_std(particles.coordinates.z, comm, buffers)
        else
            σ_E0_initial = compute_std(particles.coordinates.ΔE) # Use local std for serial
            σ_z0_initial = compute_std(particles.coordinates.z)
        end
    end

    # --- Pre-calculate Wake Parameters (if used) ---
    # Bin edges and wake constants depend on initial GLOBAL properties (sigma_z)
    # Need sigma_z0_initial (which is global in MPI mode)
    local bin_edges::AbstractRange{T} = 1.0:2.0 # Default initialization
    local bin_centers::AbstractRange{T} = (bin_edges[1:end-1] + bin_edges[2:end]) ./ 2
    local wake_factor_val::T = 0.0; local wake_sqrt_val::T = 0.0; local cτ::T = 0.0
    local n_particles_global::Int = 0
    
    if use_wakefield
        # Determine nbins from buffers (consistent across ranks)
        nbins::Int = length(buffers.λ) 
        if nbins <= 0; error("Buffer nbins invalid (must be > 0)."); end
        power_2_length = next_power_of_two(nbins * 2)
        
        if use_mpi
            # Get global particle count for wakefield
            n_local_ref = Ref(n_local)
            n_particles_global = MPI.Allreduce(n_local_ref[], MPI.SUM, comm)
            
            # Initialize MPI buffers if needed and not already initialized
            initialize_mpi_buffers!(buffers, comm_size, comm, power_2_length)
        else
            # In serial mode, global = local
            n_particles_global = n_local
        end
        
        # Define bin edges based on initial spread (use sigma_z0_initial)
        # Ensure range covers sufficient extent, e.g., +/- 7.5 sigma
        z_width = 7.5 * σ_z0_initial
        # Handle case where sigma_z is zero or very small
        if z_width < 1e-9; z_width = 1e-6; end # Use a minimum extent
        bin_edges = range(-z_width, z_width, length=nbins+1)
        bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) ./ 2
        
        # Transfer bin edges to GPU if using GPU
        if use_gpu
            transfer_bin_edges_to_gpu!(buffers.gpu_buffers, bin_edges)
        end
        
        # Wake function parameters
        kp=T(3e1); Z0=T(120π); cτ=T(4e-3);
        if cτ <= 0; error("Wake parameter cτ must be positive."); end
        if pipe_radius <= 0; error("Wake parameter pipe_radius must be positive."); end
        wake_factor_val = Z0 * SPEED_LIGHT / (π * pipe_radius)
        wake_sqrt_val = sqrt(max(T(0), 2 * kp / pipe_radius)) # Ensure non-negative argument
    end

    # --- Buffer for initial ΔE this turn (needed for E0 update) ---
    # Allocate only if E0 update is enabled, size is n_local
    ΔE_initial_turn = view(buffers.ΔE_initial_turn, 1:n_local)

    # --- Main Evolution Loop ---
    for turn in 1:n_turns

        # --- Calculate factors based on CURRENT E0 ---
        # E0 is updated at the *end* of the loop (if enabled), use current value here
        # These calculations are local but use the globally consistent E0
        local γ0 = E0 / mass
        local β0 = sqrt(max(0.0, 1.0 - 1.0 / (γ0 * γ0))) 
        local η0 = α_c - (1.0 / (γ0 * γ0)) 
        local sin_ϕs = sin(ϕs)
        local rf_factor = calc_rf_factor(freq_rf, β0)

        # --- Store Initial ΔE State (if E0 update enabled) ---
        if update_E0_flag && n_local > 0
            if use_gpu
                # Copy GPU ΔE to CPU buffer for E0 update comparison
                copyto!(ΔE_initial_turn, buffers.gpu_buffers.particle_data.ΔE)
            else
                copyto!(ΔE_initial_turn, 1, particles.coordinates.ΔE, 1, n_local)
            end
        end

        # --- Apply Physics Processes ---
        if use_gpu && n_local > 0
            # --- GPU Version of Physics Processes ---
            
            # RF Kick on GPU
            rf_kick_gpu!(buffers.gpu_buffers.particle_data, voltage, sin_ϕs, rf_factor, ϕs, gpu_config)
            
            # Quantum Excitation on GPU
            if use_excitation
                quantum_excitation_gpu!(buffers.gpu_buffers.particle_data, buffers.gpu_buffers, 
                                       E0, radius, σ_E0_initial, gpu_config)
            end
            
            # SR Damping on GPU
            if SR_damping
                synchrotron_radiation_gpu!(buffers.gpu_buffers.particle_data, E0, radius, gpu_config)
            end
            
            # Phase Advance on GPU
            if update_η
                apply_phase_advance_dynamic_gpu!(buffers.gpu_buffers.particle_data, 
                                               γ0, mass, α_c, harmonic, β0, E0, rf_factor, ϕs, gpu_config)
            else
                apply_phase_advance_gpu!(buffers.gpu_buffers.particle_data, 
                                       η0, harmonic, β0, E0, rf_factor, ϕs, gpu_config)
            end
            
            # For wakefield or E0 update with MPI, may need to transfer data back to CPU
            if use_mpi && use_wakefield
                # For wakefield with MPI, transfer data back to CPU for global operations
                transfer_particles_to_cpu!(particles, buffers.gpu_buffers.particle_data,
                                         buffers.gpu_buffers.pinned_z, buffers.gpu_buffers.pinned_ΔE)
            end
        else
            # --- CPU Version of Physics Processes ---
            
            # RF Kick on CPU
            if n_local > 0
                StochasticHaissinski.rf_kick!(voltage, sin_ϕs, rf_factor, ϕs, particles, buffers)
            end
            
            # Quantum Excitation on CPU
            if use_excitation && n_local > 0
                StochasticHaissinski.quantum_excitation!(E0, radius, σ_E0_initial, buffers, particles)
            end
            
            # SR Damping on CPU
            if SR_damping && n_local > 0
                StochasticHaissinski.synchrotron_radiation!(E0, radius, particles, buffers)
            end
            
            # Phase Advance on CPU
            if n_local > 0
                if update_η
                    # Energy-dependent slip factor (CPU)
                    @turbo for i in 1:n_local
                        Δγ_i = particles.coordinates.ΔE[i] / mass
                        # Avoid division by zero if γ0 + Δγ_i is zero or negative
                        γ_particle = γ0 + Δγ_i
                        
                        η_i = α_c - 1.0 / (γ_particle * γ_particle)
                        
                        coeff_i = (2π * harmonic * η_i / (β0 * β0 * E0))
                        
                        # Inline phase advance calculation
                        ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs) # z_to_ϕ
                        ϕ_i += coeff_i * particles.coordinates.ΔE[i]
                        particles.coordinates.z[i] = (-ϕ_i + ϕs) / rf_factor # ϕ_to_z
                    end
                else
                    # Constant slip factor η0 (CPU)
                    apply_phase_advance!(particles, η0, harmonic, β0, E0, rf_factor, ϕs)
                end
            end
        end

        # --- Wakefield Calculation ---
        if use_wakefield
            # Current bunch length for wakefield smoothing
            if use_gpu
                # Calculate current bunch length from GPU data
                if use_mpi
                    current_sigma_z_for_wake = compute_global_std_gpu(buffers.gpu_buffers.particle_data.z, comm, true)
                else
                    current_sigma_z_for_wake = compute_global_std_gpu(buffers.gpu_buffers.particle_data.z, comm, false)
                end
            else
                # Calculate from CPU data
                if use_mpi
                    current_sigma_z_for_wake = compute_global_std(particles.coordinates.z, comm, buffers)
                else
                    current_sigma_z_for_wake = compute_std(particles.coordinates.z)
                end
            end
            
            # Calculate wake current parameter
            current_γ0_wake = E0 / mass
            current_η0_wake = α_c - (1.0 / (current_γ0_wake * current_γ0_wake))
            
            # Calculate particle factor
            log10_N = log10(n_particles_global)
            floor_log10_N = log10_N >= 0 ? floor(Int, log10_N) : ceil(Int, log10_N - 1)
            power_val = max(-10, min(10, floor_log10_N))
            denominator_pf = 10.0^power_val
            if denominator_pf == 0; denominator_pf = 1.0; end
            particle_factor = (1e11 / denominator_pf) * n_particles_global
            
            # Calculate wake current
            current_wake_current = particle_factor / E0 / (2*π*radius) * current_sigma_z_for_wake / (current_η0_wake * σ_E0_initial^2)
            
            # Apply wakefield
            if use_gpu
                # GPU wakefield
                apply_wakefield_gpu!(
                    buffers.gpu_buffers.particle_data,
                    buffers.gpu_buffers,
                    wake_factor_val, wake_sqrt_val, cτ, current_wake_current,
                    current_sigma_z_for_wake,
                    gpu_config,
                    n_particles_global,
                    comm, use_mpi
                )
                
                # If MPI was used, particles were transferred back to CPU and need to go to GPU again
                if use_mpi
                    transfer_particles_to_gpu!(buffers.gpu_buffers.particle_data, particles,
                                            buffers.gpu_buffers.pinned_z, buffers.gpu_buffers.pinned_ΔE)
                end
            else
                # CPU wakefield
                apply_wakefield_inplace!(
                    particles, buffers,
                    wake_factor_val, wake_sqrt_val, cτ, current_wake_current,
                    current_sigma_z_for_wake,
                    bin_edges, bin_centers,
                    comm, use_mpi
                )
            end
        end

        # --- Global/Local E0 Update (Requires Communication in MPI) ---
        if update_E0_flag
            if use_gpu
                # Update E0 using GPU data
                E0, mean_dE_this_turn = update_E0_gpu!(
                    buffers.gpu_buffers.particle_data,
                    ΔE_initial_turn,
                    E0,
                    comm,
                    use_mpi
                )
            else
                # CPU E0 update
                local mean_dE_this_turn::T
                
                if use_mpi
                    # Calculate sum of ΔE changes on this rank
                    local sum_dE_local = zero(T)
                    if n_local > 0
                        @simd for i in 1:n_local
                            sum_dE_local += (particles.coordinates.ΔE[i] - ΔE_initial_turn[i])
                        end
                    end
                    
                    # Reduce sum of changes and local counts globally
                    reductions = MPI.Allreduce([sum_dE_local, T(n_local)], MPI.SUM, comm)
                    sum_dE_global = reductions[1]
                    n_global_count = Int(round(reductions[2]))
                    
                    # Calculate global mean energy change for this turn
                    mean_dE_this_turn = sum_dE_global / n_global_count
                    
                    # All ranks update E0 identically with the global mean change
                    E0 = E0 + mean_dE_this_turn
                    
                    # Re-center local particle energies relative to the NEW E0
                    if n_local > 0
                        safe_update_energy!(particles, mean_dE_this_turn)
                    end
                else # Serial mode
                    if n_local > 0
                        # Calculate local mean energy change (which is global in serial)
                        mean_dE_this_turn = compute_mean_diff(particles.coordinates.ΔE, ΔE_initial_turn)
                        
                        # Update E0
                        E0 = E0 + mean_dE_this_turn
                        
                        # Re-center particle energies
                        safe_update_energy!(particles, mean_dE_this_turn)
                    end
                end
            end
        end
    end # End turn loop

    # --- Final Spreads (Global if MPI) ---
    local σ_E_final::T
    local σ_z_final::T
    
    if use_gpu
        # Get final data from GPU
        if use_mpi
            # Make sure data is up to date on CPU for final spread calculation
            transfer_particles_to_cpu!(particles, buffers.gpu_buffers.particle_data,
                                     buffers.gpu_buffers.pinned_z, buffers.gpu_buffers.pinned_ΔE)
            
            # Calculate global spreads from CPU data
            σ_E_final = compute_global_std(particles.coordinates.ΔE, comm, buffers)
            σ_z_final = compute_global_std(particles.coordinates.z, comm, buffers)
        else
            # Calculate spreads directly from GPU data
            σ_E_final = compute_global_std_gpu(buffers.gpu_buffers.particle_data.ΔE, comm, false)
            σ_z_final = compute_global_std_gpu(buffers.gpu_buffers.particle_data.z, comm, false)
            
            # Transfer final state back to CPU for consistency
            transfer_particles_to_cpu!(particles, buffers.gpu_buffers.particle_data,
                                     buffers.gpu_buffers.pinned_z, buffers.gpu_buffers.pinned_ΔE)
        end
    else
        # CPU version
        if use_mpi
            σ_E_final = compute_global_std(particles.coordinates.ΔE, comm, buffers)
            σ_z_final = compute_global_std(particles.coordinates.z, comm, buffers)
        else
            σ_E_final = compute_std(particles.coordinates.ΔE)
            σ_z_final = compute_std(particles.coordinates.z)
        end
    end

    # Return final global/local spreads and the final E0 (consistent across ranks)
    return σ_E_final, σ_z_final, E0
end
