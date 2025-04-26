# File: benchmark_mpi_convolution.jl (Particle Distribution via Scatterv Strategy - Refactored)

# Ensure simulation code is accessible, adjust path as needed
include("../src/StochasticHaissinski.jl")

using .StochasticHaissinski
using BenchmarkTools
using Statistics
using LIKWID
using Dates
using Base.Threads
using Printf
using MPI # Add MPI
using Random # Import the Random module
using StructArrays # Needed for reconstructing particles StructArray

# --- Function to calculate Scatterv parameters ---
"""
Calculates particle counts and displacements for MPI Scatterv.

Args:
    n_particles_global (Int): Total number of particles.
    comm_size (Int): Number of MPI ranks.

Returns:
    Tuple{Vector{Int}, Vector{Int}}: (counts, displacements) arrays.
"""
function calculate_scatterv_params(n_particles_global::Int, comm_size::Int)
    counts = Vector{Int}(undef, comm_size)
    displs = Vector{Int}(undef, comm_size)
    base_n_local = n_particles_global ÷ comm_size
    remainder = n_particles_global % comm_size
    # Use a local variable inside the function for accumulation
    current_displacement = 0
    for r in 0:(comm_size-1)
        local_count = r < remainder ? base_n_local + 1 : base_n_local
        counts[r+1] = local_count # Julia is 1-based index
        displs[r+1] = current_displacement
        current_displacement += local_count # Modify the local accumulator
    end
    return counts, displs
end

# --- MPI Initialization ---
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)

# --- Parameters (Identical on all ranks) ---
num_threads = Threads.nthreads() # Julia threads per MPI rank
log_dir = joinpath(dirname(@__FILE__), "..", "logs/mpi_distribution") # New log dir name
parameters = Dict{String, Any}()
mkpath(log_dir) # Create directory early

# Physics parameters (defined globally)
begin
    E0_ini = 4e9; mass = MASS_ELECTRON; voltage = 5e6; harmonic = 360; radius = 250.0;
    pipe_radius = 0.00025; α_c = 3.68e-4; γ = E0_ini/mass; β = sqrt(1 - 1/γ^2);
    η = α_c - 1/γ^2; sin_ϕs = 0.5; ϕs = 5π/6;
    freq_rf = (ϕs + 10*π/180) * β * SPEED_LIGHT / (2π)
end;
# Distribution parameters (defined globally)
begin
    μ_z = 0.0; μ_E = 0.0; ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT)); σ_E0 = 1e6;
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini
end;

# --- Simulation Setup ---
n_turns = Int64(1e3)
n_particles_global = Int64(1e7); # Total number of particles

# --- Determine Local Particle Count and Scatterv Parameters (Using Function) ---
counts, displs = calculate_scatterv_params(n_particles_global, comm_size)
n_local = counts[rank+1] # This rank's local particle count

# Determine nbins based on GLOBAL particles
nbins_calc = StochasticHaissinski.next_power_of_two(max(64, Int(round(n_particles_global / 100))))

# --- Particle Generation Strategy: Rank 0 Generates, All Ranks Prepare ---
local particles # Will hold local particles after Scatterv
local E0::Float64 # All ranks need E0
local particles_global # Only defined on Rank 0

if rank == 0
    println("Rank 0 generating initial global particles...")
    Random.seed!(1234)
    particles_global, _, _, E0_generated = StochasticHaissinski.generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles_global, E0_ini, mass, ϕs, freq_rf);
    E0 = E0_generated
    println("Rank 0 finished generating.")
else
    local z_coords = Vector{Float64}(undef, n_local)
    local dE_coords = Vector{Float64}(undef, n_local)
    particles = StructArray{Particle{Float64}}((coordinates = StructArray{Coordinate{Float64}}((z=z_coords, ΔE=dE_coords)),))
    E0 = 0.0
end

# --- Broadcast E0 ---
ref_E0 = Ref(E0); MPI.Bcast!(ref_E0, 0, comm)
if rank != 0; E0 = ref_E0[]; end

# --- Scatter Particle Data ---
if rank == 0; println("Scattering particle data..."); end
# Prepare receive buffers
# Rank 0's particles variable is assigned *after* scatter
rbuf_z = MPI.Buffer(rank == 0 ? Vector{Float64}(undef, n_local) : particles.coordinates.z)
rbuf_dE = MPI.Buffer(rank == 0 ? Vector{Float64}(undef, n_local) : particles.coordinates.ΔE)
# Prepare send buffers on Rank 0
sbuf_z = nothing; sbuf_dE = nothing
if rank == 0
    sbuf_z = MPI.VBuffer(particles_global.coordinates.z, counts, displs)
    sbuf_dE = MPI.VBuffer(particles_global.coordinates.ΔE, counts, displs)
end
# Perform Scatterv
MPI.Scatterv!(sbuf_z, rbuf_z, 0, comm)
MPI.Scatterv!(sbuf_dE, rbuf_dE, 0, comm)
# Rank 0: Create its local `particles` variable from the received buffer
if rank == 0
    # rbuf_z/rbuf_dE on rank 0 now hold the scattered data
    z_coords_local = rbuf_z.data # Extract the data vector from the buffer
    dE_coords_local = rbuf_dE.data
    particles = StructArray{Particle{Float64}}((coordinates = StructArray{Coordinate{Float64}}((z=z_coords_local, ΔE=dE_coords_local)),))
    # Optional: Free large global array
    # particles_global = nothing; GC.gc()
end

MPI.Barrier(comm)
if rank == 0; println("Scatterv finished."); end

# --- Create Sim Params & Buffers ---
sim_params = SimulationParameters(E0, mass, voltage, harmonic, radius, pipe_radius, α_c, ϕs, freq_rf, n_turns, true, true, true, true, true);
buffers = StochasticHaissinski.create_simulation_buffers(n_local, nbins_calc, Float64);

# --- Pre-run / Compilation ---
if rank == 0; println("Performing pre-run for compilation..."); end
pre_params = SimulationParameters( E0, mass, voltage, harmonic, radius, pipe_radius, α_c, ϕs, freq_rf, 1, true, true, true, true, true);
particles_copy = deepcopy(particles); buffers_copy = deepcopy(buffers)
StochasticHaissinski.longitudinal_evolve!(particles_copy, pre_params, buffers_copy; show_progress=false);
MPI.Barrier(comm)
if rank == 0; println("Compilation run finished."); end

# --- Rank 0: Setup Logging ---
local log_file; local log_output # Needs to be defined before benchmark results processing
if rank == 0
    mkpath(log_dir)
    parameters["E0_ini"] = E0_ini; parameters["E0_start"] = E0; parameters["mass"] = mass
    parameters["voltage"] = voltage; parameters["harmonic"] = harmonic; parameters["radius"] = radius
    parameters["pipe_radius"] = pipe_radius; parameters["α_c"] = α_c; parameters["γ"] = γ
    parameters["β"] = β; parameters["η"] = η; parameters["sin_ϕs"] = sin_ϕs; parameters["ϕs"] = ϕs
    parameters["freq_rf"] = freq_rf; parameters["μ_z"] = μ_z; parameters["μ_E"] = μ_E
    parameters["ω_rev"] = ω_rev; parameters["σ_E0"] = σ_E0; parameters["σ_z0"] = σ_z0
    parameters["n_turns"] = n_turns; parameters["n_particles_global"] = n_particles_global
    parameters["num_threads_per_rank"] = num_threads; parameters["mpi_comm_size"] = comm_size
    parameters["nbins_calculated"] = nbins_calc
    turns_raw = @sprintf("%.0e", Float64(n_turns)); particles_raw = @sprintf("%.0e", Float64(n_particles_global))
    turns_sci = replace(replace(replace(turns_raw, "." => ""), "e+" => "e"), r"e0+" => "e")
    particles_sci = replace(replace(replace(particles_raw, "." => ""), "e+" => "e"), r"e0+" => "e")
    log_filename = "turns$(turns_sci)_particles$(particles_sci)_mpi$(comm_size)_threads$(num_threads).log"
    log_path = joinpath(log_dir, log_filename)
    log_file = open(log_path, "w") # Use simple assignment, outer var is local
    write(log_file, "=== Simulation Log (MPI Particle Scatterv) ===\n")
    write(log_file, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n"); write(log_file, "=== Parameters ===\n")
    param_names = ["E0_ini", "E0_start", "mass", "voltage", "harmonic", "radius", "pipe_radius",
                   "α_c", "γ", "β", "η", "sin_ϕs", "ϕs", "freq_rf", "μ_z", "μ_E",
                   "ω_rev", "σ_E0", "σ_z0", "n_turns", "n_particles_global",
                   "num_threads_per_rank", "mpi_comm_size", "nbins_calculated"]
    for name in param_names; value = parameters[name]; write(log_file, "$name = $value\n"); end
    write(log_file, "\n=== Simulation Output ===\n")
    # Define log_output locally within Rank 0 block
    function _log_output_rank0(args...)
        msg = string(args...); if @isdefined(log_file) && isopen(log_file); write(log_file, msg, "\n"); flush(log_file);
        else; println("Warning: log_file not available."); end
    end
    log_output = _log_output_rank0 # Assign local function to outer local var
end

# --- Benchmark Section ---
local benchmark_results # Defined outside rank check
if rank == 0; log_output("Starting benchmark..."); end
particles_bench = deepcopy(particles); buffers_bench = deepcopy(buffers)
bench_sim_params = sim_params
bench_run = @benchmarkable StochasticHaissinski.longitudinal_evolve!($particles_bench, $bench_sim_params, $buffers_bench; show_progress=false) setup=(
    particles_bench = deepcopy($particles); buffers_bench = deepcopy($buffers); MPI.Barrier($comm)
) teardown=(MPI.Barrier($comm))
results = BenchmarkTools.run(bench_run, samples=5, evals=3)
if rank == 0
    benchmark_results = results # Assign result to outer var on Rank 0
    filtered_results = median(benchmark_results); io = IOBuffer(); show(io, MIME("text/plain"), filtered_results); benchmark_details = String(take!(io))
    log_output("Benchmark results (median):\n", benchmark_details)
end

# --- Performance Monitoring Section ---
if rank == 0; log_output("\nStarting LIKWID performance monitoring..."); end
local local_flop_count::Float64 = 0.0; local perfmon_failed::Bool = false
particles_perf = deepcopy(particles); buffers_perf = deepcopy(buffers)
try
    MPI.Barrier(comm)
    _, events = @perfmon "FLOPS_DP" begin StochasticHaissinski.longitudinal_evolve!(particles_perf, bench_sim_params, buffers_perf; show_progress=false) end
    MPI.Barrier(comm)
    if haskey(events, "FLOPS_DP") && !isempty(events["FLOPS_DP"]) && haskey(first(events["FLOPS_DP"]), "RETIRED_SSE_AVX_FLOPS_ALL")
        # Still need global here because of try/catch in top-level scope
        global local_flop_count = Float64(first(events["FLOPS_DP"])["RETIRED_SSE_AVX_FLOPS_ALL"])
    else; println("Rank $rank: Warn - LIKWID count failed."); global local_flop_count = NaN; perfmon_failed = true; end
catch e
    println("Rank $rank: LIKWID failed: ", e); global local_flop_count = NaN; perfmon_failed = true; MPI.Barrier(comm)
end
# Gather FLOP counts
send_buf = Ref(local_flop_count); recv_buf = nothing
if rank == 0; recv_buf = Vector{Float64}(undef, comm_size); end
MPI.Gather!(send_buf, recv_buf, 0, comm)
# Rank 0: Process and Log LIKWID
if rank == 0
    # Use local function defined earlier for processing
    function process_likwid_results_rank0(_recv_buf, _benchmark_results)
        log_output("\nPerformance monitoring results (Aggregated Numerical):")
        total_flops_local = 0.0; any_failed_local = false; valid_flops_collected_local = 0
        if _recv_buf !== nothing
            for r in 0:(comm_size-1); flops_r = _recv_buf[r+1]
                if isnan(flops_r); log_output("Rank $r: FLOP count collection failed."); any_failed_local = true
                else; log_output("Rank $r: Local FLOPs = ", flops_r); total_flops_local += flops_r; valid_flops_collected_local += 1; end
            end
            if any_failed_local; log_output("Total FLOPs (approximate): ", total_flops_local, " from $valid_flops_collected_local ranks")
            else; log_output("Total aggregated FLOPs: ", total_flops_local); end
            if @isdefined(_benchmark_results) # Use _benchmark_results argument
                 execution_time = median(_benchmark_results.times) / 1e9
                 if execution_time > 0 && !any_failed_local; gflops_rate = total_flops_local / execution_time / 1e9; log_output("Aggregated GFLOPS rate (based on benchmark time): ", gflops_rate)
                 else; log_output("Could not calculate GFLOPS rate."); end
            else; log_output("Benchmark results N/A for GFLOPS."); end
        else; log_output("Failed to receive FLOP counts."); end
        log_output("-"^20)
    end
    # Need benchmark_results accessible here
    if @isdefined(benchmark_results)
         process_likwid_results_rank0(recv_buf, benchmark_results)
    else
         log_output("Benchmark results were not available for LIKWID processing.")
         process_likwid_results_rank0(recv_buf, nothing) # Call without results
    end
end

# --- Final Cleanup ---
if rank == 0; if @isdefined(log_file) && isopen(log_file); close(log_file); end; println("Rank 0: Log closed."); end
MPI.Finalize()