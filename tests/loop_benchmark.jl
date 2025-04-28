using BenchmarkTools
using LoopVectorization
using FLoops
using ThreadsX
using Test
using Printf
using Serialization
using Dates
using Base.Threads
using Random # For random buffers
using Pkg # To check for LoopVectorization activation

# --- 1. Define Common Data Structures ---
struct Coordinates{T<:AbstractVector}
    z::T
    ΔE::T
end

struct Particles{C<:Coordinates}
    coordinates::C
    # Add other particle properties if needed later
end

# Buffers needed across different kernels
mutable struct Buffers{T<:Real, C<:Complex{T}}
    random_buffer::Vector{T}
    indices_buffer::Vector{Int}
    weights_buffer::Vector{T}
    potential_values::Vector{T} # Input for interp_part_2
    potential_result::Vector{T} # Output for interp_part_2
    local_bin_counts::Vector{Int} # Need atomics for ThreadsX version
    global_bin_counts::Vector{Int}
    WF_temp::Vector{T}
    normalized_global_amounts::Vector{T}
    lambda_kernel::Vector{T}
    fft_W::Vector{C}
    fft_L::Vector{C}
    local_fft_W::Vector{C} # Assuming needed for local convol
    local_fft_L::Vector{C} # Assuming needed for local convol
    local_convol_freq::Vector{C}
    convol::Vector{C} # Result of IFFT
    potential_values_at_centers_global::Vector{T}
    potential::Vector{T} # Input for wakefield_last

    # Constructor to initialize with correct sizes and types
    function Buffers{T, C}(N::Int, nbins::Int) where {T<:Real, C<:Complex{T}}
        new{T, C}(
            rand(T, N),            # random_buffer
            zeros(Int, N),         # indices_buffer
            zeros(T, N),         # weights_buffer
            rand(T, nbins + 1),    # potential_values (need nbins+1 for interp)
            zeros(T, N),         # potential_result
            zeros(Int, nbins),     # local_bin_counts
            zeros(Int, nbins),     # global_bin_counts
            zeros(T, nbins),     # WF_temp
            zeros(T, nbins),     # normalized_global_amounts
            zeros(T, nbins),     # lambda_kernel
            zeros(C, nbins),     # fft_W
            zeros(C, nbins),     # fft_L
            zeros(C, nbins),     # local_fft_W (dummy init)
            zeros(C, nbins),     # local_fft_L (dummy init)
            zeros(C, nbins),     # local_convol_freq
            zeros(C, nbins),     # convol (dummy init)
            zeros(T, nbins),     # potential_values_at_centers_global
            rand(T, N)             # potential (input for last step)
        )
    end
end

# --- 2. Set up Global Dummy Data and Parameters ---
const T = Float64 # Data type
const CT = Complex{T}
const N = 100_000   # Number of particles
const N_THREADS = nthreads()
const NBINS = 1024     # Number of bins for histograms/wakefields

println("BENCHMARK SETUP:")
println("  Particles (N): $N")
println("  Threads: $N_THREADS")
println("  Bins: $NBINS")
println("-"^40)

# --- Initial Particle State ---
z_initial = rand(T, N) .* 10.0 .- 5.0
ΔE_initial = rand(T, N) .* 0.1
particles_initial = Particles(Coordinates(copy(z_initial), copy(ΔE_initial)))

# --- Buffers ---
# Initialize buffer states that might be modified
buffers_initial = Buffers{T, CT}(N, NBINS)

# --- Physics Parameters (Dummy values) ---
const rf_factor = T(0.5)
const ϕs = T(0.1)
const voltage = T(1000.0)
const sin_ϕs = sin(ϕs)
const excitation = T(0.01)
const damping_factor = T(0.999)
const coeff = T(0.001) # For Phase Advance
const value_to_subtract = T(0.005) # For safe_update_energy
const mass = T(1.0)
const γ0 = T(1000.0)
const α_c = T(1e-4)
const harmonic = 100
const β0 = sqrt(T(1.0) - T(1.0)/(γ0*γ0))
const E0 = γ0 * mass # Assuming mass is rest mass equivalent energy
const bin_start = T(-5.0)
const bin_end = T(5.0)
const bin_step = (bin_end - bin_start) / NBINS
const inv_step = T(1.0) / bin_step
const nbins_Int = NBINS # For histogram clamping
const first_edge_T = bin_start
const last_edge_T = bin_end # Assuming last edge included? Or bin_end-bin_step? Use <=
const inv_bin_step = T(1.0) / bin_step
const bin_centers = [bin_start + (i - 0.5) * bin_step for i in 1:NBINS]
const wake_factor = T(1.0)
const wake_sqrt = T(2.0)
const inv_cτ = T(0.1)
const inv_n_global = T(1.0) / N # Assuming N particles total globally
const delta_std = T(0.05)
const current = T(1.0) # For local convol

# --- Dummy Functions ---
@inline function calculate_wake_function(z, factor, sqrt_val, inv_ct)
    # Simple dummy calculation involving inputs
    return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
end

@inline function delta(z, std)
    # Simple Gaussian-like dummy
    inv_std_sqrt2pi = T(1.0) / (std * sqrt(2.0 * π))
    return inv_std_sqrt2pi * exp(-T(0.5) * (z / std)^2) * T(0.1)
end


# --- 3. Benchmark Saving Function ---
function save_benchmark_results(kernel_name::String, results::Dict, N_val::Int, N_threads_val::Int)
    log_base_dir = joinpath("logs", "benchmarks", "loop_types_benchmarks", kernel_name)
    log_filename = "$(N_val)_particles_$(N_threads_val)_threads.jl"
    raw_log_filename = "$(N_val)_particles_$(N_threads_val)_threads_raw.bin"
    log_file = joinpath(log_base_dir, log_filename)
    raw_log_file = joinpath(log_base_dir, raw_log_filename)

    println("\nSaving results for '$kernel_name' to '$log_base_dir'...")

    try
        mkpath(log_base_dir)

        # Write summary .jl file
        open(log_file, "w") do io
            println(io, "# Benchmark Results for Kernel: $kernel_name")
            println(io, "# Generated: $(Dates.now())")
            println(io, "# Number of particles (N): $N_val")
            println(io, "# Number of threads used: $N_threads_val")
            println(io, "#" * "="^60)

            order = ["Serial", "@turbo", "@floop", "ThreadsX"]
            for name in order
                if haskey(results, name)
                    trial = results[name]
                    println(io, "\n# --- Benchmark: $name ---")
                    summary_str = sprint(show, MIME("text/plain"), trial)
                    println(io, "# Summary:")
                    println(io, "#= \n$summary_str \n=#")

                    min_time_ms = minimum(trial.times) / 1e6
                    median_time_ms = median(trial.times) / 1e6
                    allocs = trial.allocs
                    memory_mib = trial.memory / (1024^2)
                    var_name_prefix = replace(name, r"[^A-Za-z0-9_]" => "_")

                    println(io, "$(var_name_prefix)_min_time_ms = $(@sprintf("%.3f", min_time_ms))")
                    println(io, "$(var_name_prefix)_median_time_ms = $(@sprintf("%.3f", median_time_ms))")
                    println(io, "$(var_name_prefix)_allocations = $allocs")
                    println(io, "$(var_name_prefix)_memory_mib = $(@sprintf("%.3f", memory_mib))")
                else
                     println(io, "\n# --- Benchmark: $name (Not Run/Available) ---")
                end
            end
            println(io, "\n# --- End of Benchmarks ---")
        end
        println("  Summary saved to: $log_file")

        # Save raw Trial objects
        try
            Serialization.serialize(raw_log_file, results)
            println("  Raw data saved to: $raw_log_file")
        catch ser_err
            @error "Failed to serialize raw benchmark data for $kernel_name!" ser_err
        end

    catch e
        @error "Failed to write benchmark log file(s) for $kernel_name!" e
    end
end

# --- 4. Define and Benchmark Each Kernel ---

# Helper function to reset state before each benchmark sample
function reset_state!(p, b, p_init, b_init, modified_fields::Tuple)
    for field in modified_fields
        if field == :ΔE
             p.coordinates.ΔE .= p_init.coordinates.ΔE
        elseif field == :z
             p.coordinates.z .= p_init.coordinates.z
        # Add buffer fields as needed, comparing field names
        elseif field == :indices_buffer
             b.indices_buffer .= b_init.indices_buffer
        elseif field == :weights_buffer
             b.weights_buffer .= b_init.weights_buffer
        elseif field == :potential_result
             b.potential_result .= b_init.potential_result
        elseif field == :local_bin_counts
             b.local_bin_counts .= b_init.local_bin_counts
        elseif field == :WF_temp
             b.WF_temp .= b_init.WF_temp
        elseif field == :normalized_global_amounts
             b.normalized_global_amounts .= b_init.normalized_global_amounts
        elseif field == :lambda_kernel
             b.lambda_kernel .= b_init.lambda_kernel
        elseif field == :fft_W
             b.fft_W .= b_init.fft_W
        elseif field == :fft_L
             b.fft_L .= b_init.fft_L
        elseif field == :local_convol_freq
             b.local_convol_freq .= b_init.local_convol_freq
         elseif field == :potential_values_at_centers_global
             b.potential_values_at_centers_global .= b_init.potential_values_at_centers_global
        # Add more buffer fields here if they are modified by kernels
        end
    end
end

# ======== KERNEL: Quantum Excitation ========
kernel_name = "quantum_excitation"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Working copy for this kernel
buffers = deepcopy(buffers_initial)     # Working copy

# --- Functions ---
function serial_quantum_excitation!(p, bufs, exc)
    ΔE_vec = p.coordinates.ΔE
    rand_buf = bufs.random_buffer
    @inbounds for i in 1:length(ΔE_vec)
        ΔE_vec[i] += exc * rand_buf[i]
    end
end
function turbo_quantum_excitation!(p, bufs, exc)
    ΔE_vec = p.coordinates.ΔE
    rand_buf = bufs.random_buffer
    @turbo for i in 1:length(ΔE_vec)
        ΔE_vec[i] += exc * rand_buf[i]
    end
end
function floop_quantum_excitation!(p, bufs, exc)
    ΔE_vec = p.coordinates.ΔE
    rand_buf = bufs.random_buffer
    let ΔE=ΔE_vec, r=rand_buf, ex=exc
        @floop ThreadedEx() for i in 1:length(ΔE)
            ΔE[i] += ex * r[i]
        end
    end
end
function threadsx_quantum_excitation!(p, bufs, exc)
    ΔE_vec = p.coordinates.ΔE
    rand_buf = bufs.random_buffer
    ThreadsX.foreach(1:length(ΔE_vec)) do i
         ΔE_vec[i] += exc * rand_buf[i]
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:ΔE,) # Fields modified by this kernel
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_quantum_excitation!(particles, buffers, $excitation) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_quantum_excitation!(particles, buffers, $excitation) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_quantum_excitation!(particles, buffers, $excitation) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_quantum_excitation!(particles, buffers, $excitation) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_quantum_excitation!(p_serial, b_serial, excitation)
    turbo_quantum_excitation!(p_turbo, b_turbo, excitation)
    floop_quantum_excitation!(p_floop, b_floop, excitation)
    threadsx_quantum_excitation!(p_tx, b_tx, excitation)
    @test p_serial.coordinates.ΔE ≈ p_turbo.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_floop.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_tx.coordinates.ΔE rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Synchrotron Radiation ========
kernel_name = "synchrotron_radiation"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Reset working copy
buffers = deepcopy(buffers_initial)

# --- Functions ---
function serial_synchrotron_radiation!(p, damp)
    ΔE_vec = p.coordinates.ΔE
    @inbounds for i in 1:length(ΔE_vec)
        ΔE_vec[i] *= damp
    end
end
function turbo_synchrotron_radiation!(p, damp)
    ΔE_vec = p.coordinates.ΔE
    @turbo for i in 1:length(ΔE_vec)
        ΔE_vec[i] *= damp
    end
end
function floop_synchrotron_radiation!(p, damp)
    ΔE_vec = p.coordinates.ΔE
    let ΔE=ΔE_vec, d=damp
        @floop ThreadedEx() for i in 1:length(ΔE)
            ΔE[i] *= d
        end
    end
end
function threadsx_synchrotron_radiation!(p, damp)
    ΔE_vec = p.coordinates.ΔE
    ThreadsX.foreach(1:length(ΔE_vec)) do i
         ΔE_vec[i] *= damp
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:ΔE,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_synchrotron_radiation!(particles, $damping_factor) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_synchrotron_radiation!(particles, $damping_factor) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_synchrotron_radiation!(particles, $damping_factor) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_synchrotron_radiation!(particles, $damping_factor) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_synchrotron_radiation!(p_serial, damping_factor)
    turbo_synchrotron_radiation!(p_turbo, damping_factor)
    floop_synchrotron_radiation!(p_floop, damping_factor)
    threadsx_synchrotron_radiation!(p_tx, damping_factor)
    @test p_serial.coordinates.ΔE ≈ p_turbo.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_floop.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_tx.coordinates.ΔE rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: RF Kick ========
kernel_name = "rf_kick"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial)

# --- Functions (Copied from previous example) ---
function serial_rf_kick!(p, rf, phi_s, volt, sin_phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @inbounds for i in 1:length(z_vec)
        ϕ_val = -z_vec[i] * rf + phi_s
        ΔE_vec[i] += volt * (sin(ϕ_val) - sin_phi_s)
    end
end
function turbo_rf_kick!(p, rf, phi_s, volt, sin_phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @turbo for i in 1:length(z_vec)
        ϕ_val = -z_vec[i] * rf + phi_s
        ΔE_vec[i] += volt * (sin(ϕ_val) - sin_phi_s)
    end
end
function floop_rf_kick!(p, rf, phi_s, volt, sin_phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    let z=z_vec, ΔE=ΔE_vec, rf=rf, phi_s=phi_s, volt=volt, sin_phi_s=sin_phi_s
        @floop ThreadedEx() for i in 1:length(z)
            ϕ_val = -z[i] * rf + phi_s
            ΔE[i] += volt * (sin(ϕ_val) - sin_phi_s)
        end
    end
end
function threadsx_rf_kick!(p, rf, phi_s, volt, sin_phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    ThreadsX.foreach(1:length(z_vec)) do i
        ϕ_val = -z_vec[i] * rf + phi_s
        ΔE_vec[i] += volt * (sin(ϕ_val) - sin_phi_s)
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:ΔE,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_rf_kick!(particles, $rf_factor, $ϕs, $voltage, $sin_ϕs) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_rf_kick!(particles, $rf_factor, $ϕs, $voltage, $sin_ϕs) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_rf_kick!(particles, $rf_factor, $ϕs, $voltage, $sin_ϕs) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_rf_kick!(particles, $rf_factor, $ϕs, $voltage, $sin_ϕs) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_rf_kick!(p_serial, rf_factor, ϕs, voltage, sin_ϕs)
    turbo_rf_kick!(p_turbo, rf_factor, ϕs, voltage, sin_ϕs)
    floop_rf_kick!(p_floop, rf_factor, ϕs, voltage, sin_ϕs)
    threadsx_rf_kick!(p_tx, rf_factor, ϕs, voltage, sin_ϕs)
    @test p_serial.coordinates.ΔE ≈ p_turbo.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_floop.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_tx.coordinates.ΔE rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Phase Advance ========
kernel_name = "phase_advance"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Modifies z, requires initial ΔE
buffers = deepcopy(buffers_initial)

# --- Functions ---
const n_local_phase_advance = N # Assume using all particles
function serial_phase_advance!(p, rf, phi_s, coef)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @inbounds for i in 1:n_local_phase_advance
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coef * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end
function turbo_phase_advance!(p, rf, phi_s, coef)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @turbo for i in 1:n_local_phase_advance
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coef * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end
function floop_phase_advance!(p, rf, phi_s, coef)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    let z=z_vec, ΔE=ΔE_vec, rf=rf, phi_s=phi_s, co=coef
        @floop ThreadedEx() for i in 1:n_local_phase_advance
            ϕ_i = -(z[i] * rf - phi_s)
            ϕ_i += co * ΔE[i]
            z[i] = (-ϕ_i + phi_s) / rf
        end
    end
end
function threadsx_phase_advance!(p, rf, phi_s, coef)
     coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
     ThreadsX.foreach(1:n_local_phase_advance) do i
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coef * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:z,) # Modifies z
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified)) # Reset z

results["Serial"]   = @benchmark serial_phase_advance!(particles, $rf_factor, $ϕs, $coeff) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_phase_advance!(particles, $rf_factor, $ϕs, $coeff) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_phase_advance!(particles, $rf_factor, $ϕs, $coeff) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_phase_advance!(particles, $rf_factor, $ϕs, $coeff) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_phase_advance!(p_serial, rf_factor, ϕs, coeff)
    turbo_phase_advance!(p_turbo, rf_factor, ϕs, coeff)
    floop_phase_advance!(p_floop, rf_factor, ϕs, coeff)
    threadsx_phase_advance!(p_tx, rf_factor, ϕs, coeff)
    @test p_serial.coordinates.z ≈ p_turbo.coordinates.z rtol=1e-12
    @test p_serial.coordinates.z ≈ p_floop.coordinates.z rtol=1e-12
    @test p_serial.coordinates.z ≈ p_tx.coordinates.z rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Safe Update Energy ========
kernel_name = "safe_update_energy"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial)

# --- Functions ---
const n_local_safe_update = N
function serial_safe_update_energy!(p, val)
    ΔE_vec = p.coordinates.ΔE
    @inbounds for i in 1:n_local_safe_update
         ΔE_vec[i] -= val
     end
end
function turbo_safe_update_energy!(p, val)
    ΔE_vec = p.coordinates.ΔE
    @turbo for i in 1:n_local_safe_update
         ΔE_vec[i] -= val
     end
end
function floop_safe_update_energy!(p, val)
    ΔE_vec = p.coordinates.ΔE
    let ΔE=ΔE_vec, v=val
        @floop ThreadedEx() for i in 1:n_local_safe_update
            ΔE[i] -= v
        end
    end
end
function threadsx_safe_update_energy!(p, val)
    ΔE_vec = p.coordinates.ΔE
    ThreadsX.foreach(1:n_local_safe_update) do i
        ΔE_vec[i] -= val
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:ΔE,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_safe_update_energy!(particles, $value_to_subtract) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_safe_update_energy!(particles, $value_to_subtract) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_safe_update_energy!(particles, $value_to_subtract) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_safe_update_energy!(particles, $value_to_subtract) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_safe_update_energy!(p_serial, value_to_subtract)
    turbo_safe_update_energy!(p_turbo, value_to_subtract)
    floop_safe_update_energy!(p_floop, value_to_subtract)
    threadsx_safe_update_energy!(p_tx, value_to_subtract)
    @test p_serial.coordinates.ΔE ≈ p_turbo.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_floop.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_tx.coordinates.ΔE rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Update Eta ========
kernel_name = "update_eta"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Modifies z, requires initial ΔE
buffers = deepcopy(buffers_initial)

# --- Functions ---
const n_local_update_eta = N
function serial_update_eta!(p, m, g0, ac, h, b0, e0_phys, rf, phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @inbounds for i in 1:n_local_update_eta
        Δγ_i = ΔE_vec[i] / m
        γ_particle = g0 + Δγ_i
        η_i = ac - T(1.0) / (γ_particle * γ_particle) # Avoid division by zero handled implicitly if γ_particle stays positive
        coeff_i = (T(2.0 * π) * h * η_i / (b0 * b0 * e0_phys))
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coeff_i * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end
function turbo_update_eta!(p, m, g0, ac, h, b0, e0_phys, rf, phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    @turbo for i in 1:n_local_update_eta
        Δγ_i = ΔE_vec[i] / m
        γ_particle = g0 + Δγ_i
        η_i = ac - T(1.0) / (γ_particle * γ_particle)
        coeff_i = (T(2.0 * π) * h * η_i / (b0 * b0 * e0_phys))
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coeff_i * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end
function floop_update_eta!(p, m, g0, ac, h, b0, e0_phys, rf, phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    let z=z_vec, ΔE=ΔE_vec, m=m, g0=g0, ac=ac, h=h, b0=b0, e0=e0_phys, rf=rf, phi_s=phi_s
        @floop ThreadedEx() for i in 1:n_local_update_eta
            Δγ_i = ΔE[i] / m
            γ_particle = g0 + Δγ_i
            η_i = ac - T(1.0) / (γ_particle * γ_particle)
            coeff_i = (T(2.0 * π) * h * η_i / (b0 * b0 * e0))
            ϕ_i = -(z[i] * rf - phi_s)
            ϕ_i += coeff_i * ΔE[i]
            z[i] = (-ϕ_i + phi_s) / rf
        end
    end
end
function threadsx_update_eta!(p, m, g0, ac, h, b0, e0_phys, rf, phi_s)
    coords = p.coordinates; z_vec = coords.z; ΔE_vec = coords.ΔE
    ThreadsX.foreach(1:n_local_update_eta) do i
        Δγ_i = ΔE_vec[i] / m
        γ_particle = g0 + Δγ_i
        η_i = ac - T(1.0) / (γ_particle * γ_particle)
        coeff_i = (T(2.0 * π) * h * η_i / (b0 * b0 * e0_phys))
        ϕ_i = -(z_vec[i] * rf - phi_s)
        ϕ_i += coeff_i * ΔE_vec[i]
        z_vec[i] = (-ϕ_i + phi_s) / rf
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:z,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_update_eta!(particles, $mass, $γ0, $α_c, $harmonic, $β0, $E0, $rf_factor, $ϕs) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_update_eta!(particles, $mass, $γ0, $α_c, $harmonic, $β0, $E0, $rf_factor, $ϕs) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_update_eta!(particles, $mass, $γ0, $α_c, $harmonic, $β0, $E0, $rf_factor, $ϕs) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_update_eta!(particles, $mass, $γ0, $α_c, $harmonic, $β0, $E0, $rf_factor, $ϕs) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_update_eta!(p_serial, mass, γ0, α_c, harmonic, β0, E0, rf_factor, ϕs)
    turbo_update_eta!(p_turbo, mass, γ0, α_c, harmonic, β0, E0, rf_factor, ϕs)
    floop_update_eta!(p_floop, mass, γ0, α_c, harmonic, β0, E0, rf_factor, ϕs)
    threadsx_update_eta!(p_tx, mass, γ0, α_c, harmonic, β0, E0, rf_factor, ϕs)
    @test p_serial.coordinates.z ≈ p_turbo.coordinates.z rtol=1e-12
    @test p_serial.coordinates.z ≈ p_floop.coordinates.z rtol=1e-12
    @test p_serial.coordinates.z ≈ p_tx.coordinates.z rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Zero Alloc Interp (Part 1) ========
kernel_name = "zero_alloc_interp_part_1"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Uses z
buffers = deepcopy(buffers_initial)     # Modifies indices_buffer, weights_buffer

# --- Functions ---
const particle_positions_zaip1 = particles.coordinates.z # Alias for clarity
function serial_zero_alloc_interp_p1!(idxs, wgts, pos, bstart, bend, invstep, nbins)
    @inbounds for i in 1:length(pos)
        p = pos[i]
        normalized_pos = (clamp(p, bstart, bend) - bstart) * invstep
        base_idx = floor(Int, normalized_pos) + 1
        idxs[i] = clamp(base_idx, 1, nbins) # Clamp to ensure index is valid for part 2
        wgts[i] = normalized_pos - floor(normalized_pos)
    end
end
function turbo_zero_alloc_interp_p1!(idxs, wgts, pos, bstart, bend, invstep, nbins)
    @turbo for i in 1:length(pos)
        p = pos[i]
        normalized_pos = (clamp(p, bstart, bend) - bstart) * invstep
        base_idx = floor(Int, normalized_pos) + 1
        idxs[i] = clamp(base_idx, 1, nbins)
        wgts[i] = normalized_pos - floor(normalized_pos)
    end
end
function floop_zero_alloc_interp_p1!(idxs, wgts, pos, bstart, bend, invstep, nbins)
    let idxs=idxs, wgts=wgts, pos=pos, bstart=bstart, bend=bend, invstep=invstep, nbins=nbins
        @floop ThreadedEx() for i in 1:length(pos)
            p = pos[i]
            normalized_pos = (clamp(p, bstart, bend) - bstart) * invstep
            base_idx = floor(Int, normalized_pos) + 1
            idxs[i] = clamp(base_idx, 1, nbins)
            wgts[i] = normalized_pos - floor(normalized_pos)
        end
    end
end
function threadsx_zero_alloc_interp_p1!(idxs, wgts, pos, bstart, bend, invstep, nbins)
     ThreadsX.foreach(1:length(pos)) do i
        p = pos[i]
        normalized_pos = (clamp(p, bstart, bend) - bstart) * invstep
        base_idx = floor(Int, normalized_pos) + 1
        idxs[i] = clamp(base_idx, 1, nbins)
        wgts[i] = normalized_pos - floor(normalized_pos)
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:indices_buffer, :weights_buffer)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_zero_alloc_interp_p1!(buffers.indices_buffer, buffers.weights_buffer, $particle_positions_zaip1, $bin_start, $bin_end, $inv_step, $NBINS) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_zero_alloc_interp_p1!(buffers.indices_buffer, buffers.weights_buffer, $particle_positions_zaip1, $bin_start, $bin_end, $inv_step, $NBINS) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_zero_alloc_interp_p1!(buffers.indices_buffer, buffers.weights_buffer, $particle_positions_zaip1, $bin_start, $bin_end, $inv_step, $NBINS) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_zero_alloc_interp_p1!(buffers.indices_buffer, buffers.weights_buffer, $particle_positions_zaip1, $bin_start, $bin_end, $inv_step, $NBINS) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_zero_alloc_interp_p1!(b_serial.indices_buffer, b_serial.weights_buffer, particle_positions_zaip1, bin_start, bin_end, inv_step, NBINS)
    turbo_zero_alloc_interp_p1!(b_turbo.indices_buffer, b_turbo.weights_buffer, particle_positions_zaip1, bin_start, bin_end, inv_step, NBINS)
    floop_zero_alloc_interp_p1!(b_floop.indices_buffer, b_floop.weights_buffer, particle_positions_zaip1, bin_start, bin_end, inv_step, NBINS)
    threadsx_zero_alloc_interp_p1!(b_tx.indices_buffer, b_tx.weights_buffer, particle_positions_zaip1, bin_start, bin_end, inv_step, NBINS)
    # Test both modified buffers
    @test b_serial.indices_buffer ≈ b_turbo.indices_buffer
    @test b_serial.indices_buffer ≈ b_floop.indices_buffer
    @test b_serial.indices_buffer ≈ b_tx.indices_buffer
    @test b_serial.weights_buffer ≈ b_turbo.weights_buffer rtol=1e-12
    @test b_serial.weights_buffer ≈ b_floop.weights_buffer rtol=1e-12
    @test b_serial.weights_buffer ≈ b_tx.weights_buffer rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Zero Alloc Interp (Part 2) ========
kernel_name = "zero_alloc_interp_part_2"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial)
# --- Ensure Part 1 has run on the initial buffers for consistent input ---
serial_zero_alloc_interp_p1!(buffers_initial.indices_buffer, buffers_initial.weights_buffer, particle_positions_zaip1, bin_start, bin_end, inv_step, NBINS)
# --- Functions ---
function serial_zero_alloc_interp_p2!(res, idxs, wgts, vals)
    @inbounds for i in 1:length(wgts) # Loop over particles
        idx = idxs[i]
        w = wgts[i]
        v1 = vals[idx]
        v2 = vals[idx+1]
        res[i] = v1 * (T(1.0) - w) + v2 * w
    end
end
function turbo_zero_alloc_interp_p2!(res, idxs, wgts, vals)
     @turbo for i in 1:length(wgts)
        idx = idxs[i]
        w = wgts[i]
        v1 = vals[idx]
        v2 = vals[idx+1]
        res[i] = v1 * (T(1.0) - w) + v2 * w
    end
end
# function floop_zero_alloc_interp_p2!(res, idxs, wgts, vals)
#     let res=res, idxs=idxs, wgts=wgts, vals=vals
#         @floop ThreadedEx() for i in 1:length(wgts)
#             idx = idxs[i]
#             w = wgts[i]
#             v1 = vals[idx]
#             v2 = vals[idx+1]
#             res[i] = v1 * (1.0 - w) + v2 * w
#         end
#     end
# end
# function threadsx_zero_alloc_interp_p2!(res, idxs, wgts, vals)
#     ThreadsX.foreach(1:length(wgts)) do i
#         idx = idxs[i]
#         w = wgts[i]
#         v1 = vals[idx]
#         v2 = vals[idx+1]
#         res[i] = v1 * (T(1.0) - w) + v2 * w
#     end
# end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:potential_result,) # Modifies the result buffer
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_zero_alloc_interp_p2!(buffers.potential_result, buffers.indices_buffer, buffers.weights_buffer, buffers.potential_values) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_zero_alloc_interp_p2!(buffers.potential_result, buffers.indices_buffer, buffers.weights_buffer, buffers.potential_values) setup=($setup_ex) evals=1
# results["@floop"]   = @benchmark floop_zero_alloc_interp_p2!(buffers.potential_result, buffers.indices_buffer, buffers.weights_buffer, buffers.potential_values) setup=($setup_ex) evals=1
# results["ThreadsX"] = @benchmark threadsx_zero_alloc_interp_p2!(buffers.potential_result, buffers.indices_buffer, buffers.weights_buffer, buffers.potential_values) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    # p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    # p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Use the initial buffers where part 1 was run
    serial_zero_alloc_interp_p2!(b_serial.potential_result, buffers_initial.indices_buffer, buffers_initial.weights_buffer, buffers_initial.potential_values)
    turbo_zero_alloc_interp_p2!(b_turbo.potential_result, buffers_initial.indices_buffer, buffers_initial.weights_buffer, buffers_initial.potential_values)
    # floop_zero_alloc_interp_p2!(b_floop.potential_result, buffers_initial.indices_buffer, buffers_initial.weights_buffer, buffers_initial.potential_values)
    # threadsx_zero_alloc_interp_p2!(b_tx.potential_result, buffers_initial.indices_buffer, buffers_initial.weights_buffer, buffers_initial.potential_values)
    @test b_serial.potential_result ≈ b_turbo.potential_result rtol=1e-12
    # @test b_serial.potential_result ≈ b_floop.potential_result rtol=1e-12
    # @test b_serial.potential_result ≈ b_tx.potential_result rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Calculate Wake Function ========
kernel_name = "calc_wake_func"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial)     # Modifies WF_temp

# --- Functions ---
function serial_calc_wake_func!(wf_temp, centers, factor, sqrt_val, inv_ct)
    @inbounds for i in 1:length(centers) # Iterate over bins
        wf_temp[i] = calculate_wake_function(centers[i], factor, sqrt_val, inv_ct)
    end
end
function turbo_calc_wake_func!(wf_temp, centers, factor, sqrt_val, inv_ct)
    @turbo for i in 1:length(centers)
        wf_temp[i] = calculate_wake_function(centers[i], factor, sqrt_val, inv_ct)
    end
end
function floop_calc_wake_func!(wf_temp, centers, factor, sqrt_val, inv_ct)
    let wf=wf_temp, ce=centers, fa=factor, sq=sqrt_val, ic=inv_ct
        @floop ThreadedEx() for i in 1:length(ce)
             wf[i] = calculate_wake_function(ce[i], fa, sq, ic)
        end
    end
end
function threadsx_calc_wake_func!(wf_temp, centers, factor, sqrt_val, inv_ct)
    ThreadsX.foreach(1:length(centers)) do i
        wf_temp[i] = calculate_wake_function(centers[i], factor, sqrt_val, inv_ct)
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:WF_temp,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_calc_wake_func!(buffers.WF_temp, $bin_centers, $wake_factor, $wake_sqrt, $inv_cτ) setup=($setup_ex) evals=1
# results["@turbo"]   = @benchmark turbo_calc_wake_func!(buffers.WF_temp, $bin_centers, $wake_factor, $wake_sqrt, $inv_cτ) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_calc_wake_func!(buffers.WF_temp, $bin_centers, $wake_factor, $wake_sqrt, $inv_cτ) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_calc_wake_func!(buffers.WF_temp, $bin_centers, $wake_factor, $wake_sqrt, $inv_cτ) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    # p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_calc_wake_func!(b_serial.WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ)
    # turbo_calc_wake_func!(b_turbo.WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ)
    floop_calc_wake_func!(b_floop.WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ)
    threadsx_calc_wake_func!(b_tx.WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ)
    # @test b_serial.WF_temp ≈ b_turbo.WF_temp rtol=1e-12
    @test b_serial.WF_temp ≈ b_floop.WF_temp rtol=1e-12
    @test b_serial.WF_temp ≈ b_tx.WF_temp rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Norm Global ========
kernel_name = "wakefield_norm_global"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial) # Modifies normalized_global_amounts, needs global_bin_counts

# --- Ensure global_bin_counts has some data (e.g., from a dummy histogram run) ---
buffers_initial.global_bin_counts .= rand(0:N÷NBINS, NBINS) # Dummy counts

# --- Functions ---
function serial_wakefield_norm_global!(norm_amnts, counts, inv_n)
    @inbounds for i in 1:length(counts) # Iterate over bins
        norm_amnts[i] = counts[i] * inv_n
    end
end
function turbo_wakefield_norm_global!(norm_amnts, counts, inv_n)
    @turbo for i in 1:length(counts)
        norm_amnts[i] = counts[i] * inv_n
    end
end
function floop_wakefield_norm_global!(norm_amnts, counts, inv_n)
    let na=norm_amnts, co=counts, invn=inv_n
        @floop ThreadedEx() for i in 1:length(co)
             na[i] = co[i] * invn
        end
    end
end
function threadsx_wakefield_norm_global!(norm_amnts, counts, inv_n)
    ThreadsX.foreach(1:length(counts)) do i
        norm_amnts[i] = counts[i] * inv_n
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:normalized_global_amounts,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified); buffers.global_bin_counts .= $buffers_initial.global_bin_counts) # Reset norm_amounts, ensure counts are reset too

results["Serial"]   = @benchmark serial_wakefield_norm_global!(buffers.normalized_global_amounts, buffers.global_bin_counts, $inv_n_global) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_wakefield_norm_global!(buffers.normalized_global_amounts, buffers.global_bin_counts, $inv_n_global) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_norm_global!(buffers.normalized_global_amounts, buffers.global_bin_counts, $inv_n_global) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_norm_global!(buffers.normalized_global_amounts, buffers.global_bin_counts, $inv_n_global) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Ensure counts are set in test buffers
    b_serial.global_bin_counts .= buffers_initial.global_bin_counts
    b_turbo.global_bin_counts .= buffers_initial.global_bin_counts
    b_floop.global_bin_counts .= buffers_initial.global_bin_counts
    b_tx.global_bin_counts .= buffers_initial.global_bin_counts

    serial_wakefield_norm_global!(b_serial.normalized_global_amounts, b_serial.global_bin_counts, inv_n_global)
    turbo_wakefield_norm_global!(b_turbo.normalized_global_amounts, b_turbo.global_bin_counts, inv_n_global)
    floop_wakefield_norm_global!(b_floop.normalized_global_amounts, b_floop.global_bin_counts, inv_n_global)
    threadsx_wakefield_norm_global!(b_tx.normalized_global_amounts, b_tx.global_bin_counts, inv_n_global)
    @test b_serial.normalized_global_amounts ≈ b_turbo.normalized_global_amounts rtol=1e-12
    @test b_serial.normalized_global_amounts ≈ b_floop.normalized_global_amounts rtol=1e-12
    @test b_serial.normalized_global_amounts ≈ b_tx.normalized_global_amounts rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Delta ========
kernel_name = "wakefield_delta"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial) # Modifies lambda_kernel

# --- Functions ---
function serial_wakefield_delta!(lambda, centers, std)
    @inbounds for i in 1:length(centers) # Iterate over bins
        lambda[i] = delta(centers[i], std)
    end
end
function turbo_wakefield_delta!(lambda, centers, std)
    @turbo for i in 1:length(centers)
        lambda[i] = delta(centers[i], std)
    end
end
function floop_wakefield_delta!(lambda, centers, std)
    let la=lambda, ce=centers, s=std
        @floop ThreadedEx() for i in 1:length(ce)
             la[i] = delta(ce[i], s)
        end
    end
end
function threadsx_wakefield_delta!(lambda, centers, std)
    ThreadsX.foreach(1:length(centers)) do i
        lambda[i] = delta(centers[i], std)
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:lambda_kernel,)
setup_ex = :(reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified))

results["Serial"]   = @benchmark serial_wakefield_delta!(buffers.lambda_kernel, $bin_centers, $delta_std) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_wakefield_delta!(buffers.lambda_kernel, $bin_centers, $delta_std) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_delta!(buffers.lambda_kernel, $bin_centers, $delta_std) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_delta!(buffers.lambda_kernel, $bin_centers, $delta_std) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    serial_wakefield_delta!(b_serial.lambda_kernel, bin_centers, delta_std)
    turbo_wakefield_delta!(b_turbo.lambda_kernel, bin_centers, delta_std)
    floop_wakefield_delta!(b_floop.lambda_kernel, bin_centers, delta_std)
    threadsx_wakefield_delta!(b_tx.lambda_kernel, bin_centers, delta_std)
    @test b_serial.lambda_kernel ≈ b_turbo.lambda_kernel rtol=1e-12
    @test b_serial.lambda_kernel ≈ b_floop.lambda_kernel rtol=1e-12
    @test b_serial.lambda_kernel ≈ b_tx.lambda_kernel rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Complex ========
kernel_name = "wakefield_complex"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial) # Modifies fft_W, fft_L. Needs WF_temp, lambda_kernel, normalized_global_amounts

# --- Ensure dependency buffers have data ---
serial_calc_wake_func!(buffers_initial.WF_temp, bin_centers, wake_factor, wake_sqrt, inv_cτ)
serial_wakefield_delta!(buffers_initial.lambda_kernel, bin_centers, delta_std)
buffers_initial.global_bin_counts .= rand(0:N÷NBINS, NBINS) # Dummy counts for norm
serial_wakefield_norm_global!(buffers_initial.normalized_global_amounts, buffers_initial.global_bin_counts, inv_n_global)

# --- Functions ---
function serial_wakefield_complex!(fw, fl, wf, lk, na)
    @inbounds for i in 1:length(wf) # Iterate over bins
        fw[i] = Complex{T}(wf[i])
        fl[i] = Complex{T}(lk[i] * na[i])
    end
end
# function turbo_wakefield_complex!(fw, fl, wf, lk, na)
#     # @turbo might struggle with complex construction directly, test it
#     @turbo for i in 1:length(wf)
#         fw[i] = Complex{T}(wf[i])
#         fl[i] = Complex{T}(lk[i] * na[i])
#     end
# end
function floop_wakefield_complex!(fw, fl, wf, lk, na)
    let fw=fw, fl=fl, wf=wf, lk=lk, na=na
        @floop ThreadedEx() for i in 1:length(wf)
             fw[i] = Complex{T}(wf[i])
             fl[i] = Complex{T}(lk[i] * na[i])
        end
    end
end
function threadsx_wakefield_complex!(fw, fl, wf, lk, na)
    ThreadsX.foreach(1:length(wf)) do i
        fw[i] = Complex{T}(wf[i])
        fl[i] = Complex{T}(lk[i] * na[i])
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:fft_W, :fft_L)
setup_ex = quote # Multi-line setup
    reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified)
    # Ensure input buffers have initial state for benchmark run
    buffers.WF_temp .= $buffers_initial.WF_temp
    buffers.lambda_kernel .= $buffers_initial.lambda_kernel
    buffers.normalized_global_amounts .= $buffers_initial.normalized_global_amounts
end

results["Serial"]   = @benchmark serial_wakefield_complex!(buffers.fft_W, buffers.fft_L, buffers.WF_temp, buffers.lambda_kernel, buffers.normalized_global_amounts) setup=($setup_ex) evals=1
# results["@turbo"]   = @benchmark turbo_wakefield_complex!(buffers.fft_W, buffers.fft_L, buffers.WF_temp, buffers.lambda_kernel, buffers.normalized_global_amounts) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_complex!(buffers.fft_W, buffers.fft_L, buffers.WF_temp, buffers.lambda_kernel, buffers.normalized_global_amounts) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_complex!(buffers.fft_W, buffers.fft_L, buffers.WF_temp, buffers.lambda_kernel, buffers.normalized_global_amounts) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    # p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Set inputs for test buffers
    b_serial.WF_temp .= buffers_initial.WF_temp; b_serial.lambda_kernel .= buffers_initial.lambda_kernel; b_serial.normalized_global_amounts .= buffers_initial.normalized_global_amounts;
    # b_turbo.WF_temp .= buffers_initial.WF_temp; b_turbo.lambda_kernel .= buffers_initial.lambda_kernel; b_turbo.normalized_global_amounts .= buffers_initial.normalized_global_amounts;
    b_floop.WF_temp .= buffers_initial.WF_temp; b_floop.lambda_kernel .= buffers_initial.lambda_kernel; b_floop.normalized_global_amounts .= buffers_initial.normalized_global_amounts;
    b_tx.WF_temp .= buffers_initial.WF_temp; b_tx.lambda_kernel .= buffers_initial.lambda_kernel; b_tx.normalized_global_amounts .= buffers_initial.normalized_global_amounts;

    serial_wakefield_complex!(b_serial.fft_W, b_serial.fft_L, b_serial.WF_temp, b_serial.lambda_kernel, b_serial.normalized_global_amounts)
    # turbo_wakefield_complex!(b_turbo.fft_W, b_turbo.fft_L, b_turbo.WF_temp, b_turbo.lambda_kernel, b_turbo.normalized_global_amounts)
    floop_wakefield_complex!(b_floop.fft_W, b_floop.fft_L, b_floop.WF_temp, b_floop.lambda_kernel, b_floop.normalized_global_amounts)
    threadsx_wakefield_complex!(b_tx.fft_W, b_tx.fft_L, b_tx.WF_temp, b_tx.lambda_kernel, b_tx.normalized_global_amounts)
    # @test b_serial.fft_W ≈ b_turbo.fft_W rtol=1e-12
    @test b_serial.fft_W ≈ b_floop.fft_W rtol=1e-12
    @test b_serial.fft_W ≈ b_tx.fft_W rtol=1e-12
    # @test b_serial.fft_L ≈ b_turbo.fft_L rtol=1e-12
    @test b_serial.fft_L ≈ b_floop.fft_L rtol=1e-12
    @test b_serial.fft_L ≈ b_tx.fft_L rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Local Convol ========
kernel_name = "wakefield_local_convol"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial) # Modifies local_convol_freq, Needs local_fft_W, local_fft_L

# --- Ensure dependency buffers have data (dummy) ---
buffers_initial.local_fft_W .= rand(CT, NBINS)
buffers_initial.local_fft_L .= rand(CT, NBINS)

# --- Functions ---
const chunk_size_convol = NBINS # Assume chunk_size is nbins
function serial_wakefield_local_convol!(lcf, lfw, lfl, curr)
    @inbounds @simd for i in 1:chunk_size_convol # @simd is hint for serial compiler
        lcf[i] = lfw[i] * lfl[i] * curr
    end
end
function turbo_wakefield_local_convol!(lcf, lfw, lfl, curr)
    @turbo for i in 1:chunk_size_convol
        lcf[i] = lfw[i] * lfl[i] * curr
    end
end
function floop_wakefield_local_convol!(lcf, lfw, lfl, curr)
    let lcf=lcf, lfw=lfw, lfl=lfl, c=curr
        @floop ThreadedEx() for i in 1:chunk_size_convol
             lcf[i] = lfw[i] * lfl[i] * c
        end
    end
end
function threadsx_wakefield_local_convol!(lcf, lfw, lfl, curr)
    ThreadsX.foreach(1:chunk_size_convol) do i
        lcf[i] = lfw[i] * lfl[i] * curr
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:local_convol_freq,)
setup_ex = quote
    reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified)
    buffers.local_fft_W .= $buffers_initial.local_fft_W
    buffers.local_fft_L .= $buffers_initial.local_fft_L
end

results["Serial"]   = @benchmark serial_wakefield_local_convol!(buffers.local_convol_freq, buffers.local_fft_W, buffers.local_fft_L, $current) setup=($setup_ex) evals=1
# results["@turbo"]   = @benchmark turbo_wakefield_local_convol!(buffers.local_convol_freq, buffers.local_fft_W, buffers.local_fft_L, $current) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_local_convol!(buffers.local_convol_freq, buffers.local_fft_W, buffers.local_fft_L, $current) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_local_convol!(buffers.local_convol_freq, buffers.local_fft_W, buffers.local_fft_L, $current) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    # p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Set inputs
    b_serial.local_fft_W .= buffers_initial.local_fft_W; b_serial.local_fft_L .= buffers_initial.local_fft_L;
    # b_turbo.local_fft_W .= buffers_initial.local_fft_W; b_turbo.local_fft_L .= buffers_initial.local_fft_L;
    b_floop.local_fft_W .= buffers_initial.local_fft_W; b_floop.local_fft_L .= buffers_initial.local_fft_L;
    b_tx.local_fft_W .= buffers_initial.local_fft_W; b_tx.local_fft_L .= buffers_initial.local_fft_L;

    serial_wakefield_local_convol!(b_serial.local_convol_freq, b_serial.local_fft_W, b_serial.local_fft_L, current)
    # turbo_wakefield_local_convol!(b_turbo.local_convol_freq, b_turbo.local_fft_W, b_turbo.local_fft_L, current)
    floop_wakefield_local_convol!(b_floop.local_convol_freq, b_floop.local_fft_W, b_floop.local_fft_L, current)
    threadsx_wakefield_local_convol!(b_tx.local_convol_freq, b_tx.local_fft_W, b_tx.local_fft_L, current)
    # @test b_serial.local_convol_freq ≈ b_turbo.local_convol_freq rtol=1e-12
    @test b_serial.local_convol_freq ≈ b_floop.local_convol_freq rtol=1e-12
    @test b_serial.local_convol_freq ≈ b_tx.local_convol_freq rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Real ========
kernel_name = "wakefield_real"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial)
buffers = deepcopy(buffers_initial) # Modifies pot..., needs convol

# --- Ensure convol has data ---
buffers_initial.convol .= rand(CT, NBINS) # Dummy IFFT result

# --- Functions ---
function serial_wakefield_real!(pot_vals, conv)
    @inbounds for i in 1:length(conv) # Iterate over bins
        pot_vals[i] = real(conv[i])
    end
end
function turbo_wakefield_real!(pot_vals, conv)
    @turbo for i in 1:length(conv)
        pot_vals[i] = real(conv[i])
    end
end
function floop_wakefield_real!(pot_vals, conv)
    let pv=pot_vals, co=conv
        @floop ThreadedEx() for i in 1:length(co)
             pv[i] = real(co[i])
        end
    end
end
function threadsx_wakefield_real!(pot_vals, conv)
    ThreadsX.foreach(1:length(conv)) do i
        pot_vals[i] = real(conv[i])
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:potential_values_at_centers_global,)
setup_ex = quote
    reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified)
    buffers.convol .= $buffers_initial.convol
end

results["Serial"]   = @benchmark serial_wakefield_real!(buffers.potential_values_at_centers_global, buffers.convol) setup=($setup_ex) evals=1
# results["@turbo"]   = @benchmark turbo_wakefield_real!(buffers.potential_values_at_centers_global, buffers.convol) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_real!(buffers.potential_values_at_centers_global, buffers.convol) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_real!(buffers.potential_values_at_centers_global, buffers.convol) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    # p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Set inputs
    b_serial.convol .= buffers_initial.convol;
    # b_turbo.convol .= buffers_initial.convol;
    b_floop.convol .= buffers_initial.convol;
    b_tx.convol .= buffers_initial.convol;

    serial_wakefield_real!(b_serial.potential_values_at_centers_global, b_serial.convol)
    # turbo_wakefield_real!(b_turbo.potential_values_at_centers_global, b_turbo.convol)
    floop_wakefield_real!(b_floop.potential_values_at_centers_global, b_floop.convol)
    threadsx_wakefield_real!(b_tx.potential_values_at_centers_global, b_tx.convol)
    # @test b_serial.potential_values_at_centers_global ≈ b_turbo.potential_values_at_centers_global rtol=1e-12
    @test b_serial.potential_values_at_centers_global ≈ b_floop.potential_values_at_centers_global rtol=1e-12
    @test b_serial.potential_values_at_centers_global ≈ b_tx.potential_values_at_centers_global rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


# ======== KERNEL: Wakefield Last ========
kernel_name = "wakefield_last"
println("\n" * "="^10 * " KERNEL: $kernel_name " * "="^10)
particles = deepcopy(particles_initial) # Modifies ΔE
buffers = deepcopy(buffers_initial)     # Needs potential buffer

# --- Ensure potential buffer has data ---
buffers_initial.potential .= rand(T, N) # Dummy potential values

# --- Functions ---
const n_particles_last = N
function serial_wakefield_last!(p, pot_buf)
    ΔE_vec = p.coordinates.ΔE
    @inbounds for i in 1:n_particles_last
        ΔE_vec[i] -= pot_buf[i]
    end
end
function turbo_wakefield_last!(p, pot_buf)
    ΔE_vec = p.coordinates.ΔE
    @turbo for i in 1:n_particles_last
        ΔE_vec[i] -= pot_buf[i]
    end
end
function floop_wakefield_last!(p, pot_buf)
    ΔE_vec = p.coordinates.ΔE
    let ΔE=ΔE_vec, pb=pot_buf
        @floop ThreadedEx() for i in 1:n_particles_last
            ΔE[i] -= pb[i]
        end
    end
end
function threadsx_wakefield_last!(p, pot_buf)
    ΔE_vec = p.coordinates.ΔE
    ThreadsX.foreach(1:n_particles_last) do i
        ΔE_vec[i] -= pot_buf[i]
    end
end

# --- Benchmarking ---
results = Dict{String, BenchmarkTools.Trial}()
modified = (:ΔE,)
setup_ex = quote
    reset_state!(particles, buffers, $particles_initial, $buffers_initial, $modified)
    buffers.potential .= $buffers_initial.potential
end

results["Serial"]   = @benchmark serial_wakefield_last!(particles, buffers.potential) setup=($setup_ex) evals=1
results["@turbo"]   = @benchmark turbo_wakefield_last!(particles, buffers.potential) setup=($setup_ex) evals=1
results["@floop"]   = @benchmark floop_wakefield_last!(particles, buffers.potential) setup=($setup_ex) evals=1
results["ThreadsX"] = @benchmark threadsx_wakefield_last!(particles, buffers.potential) setup=($setup_ex) evals=1

# --- Verification ---
try
    p_serial = deepcopy(particles_initial); b_serial = deepcopy(buffers_initial)
    p_turbo = deepcopy(particles_initial); b_turbo = deepcopy(buffers_initial)
    p_floop = deepcopy(particles_initial); b_floop = deepcopy(buffers_initial)
    p_tx = deepcopy(particles_initial); b_tx = deepcopy(buffers_initial)
    # Set inputs
    b_serial.potential .= buffers_initial.potential;
    b_turbo.potential .= buffers_initial.potential;
    b_floop.potential .= buffers_initial.potential;
    b_tx.potential .= buffers_initial.potential;

    serial_wakefield_last!(p_serial, b_serial.potential)
    turbo_wakefield_last!(p_turbo, b_turbo.potential)
    floop_wakefield_last!(p_floop, b_floop.potential)
    threadsx_wakefield_last!(p_tx, b_tx.potential)
    @test p_serial.coordinates.ΔE ≈ p_turbo.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_floop.coordinates.ΔE rtol=1e-12
    @test p_serial.coordinates.ΔE ≈ p_tx.coordinates.ΔE rtol=1e-12
    println("Verification PASSED for $kernel_name")
catch e
    println("ERROR during verification for $kernel_name: $e")
end

# --- Save Results ---
save_benchmark_results(kernel_name, results, N, N_THREADS)


println("\n" * "="^40)
println("All kernel benchmarks finished.")
println("Results saved in logs/benchmarks/loop_types_benchmarks/<kernel_name>/")
println("="^40)