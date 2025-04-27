# benchmark_total.jl
# Unified benchmark script for StochasticHaissinski simulation
# Supports serial and MPI execution modes selected via command-line argument.

# Determine the base path relative to the script's location

script_dir = dirname(@__FILE__)
project_root = joinpath(script_dir, "..")
src_path = joinpath(project_root, "src", "StochasticHaissinski.jl")
include(src_path)

# include("../src/StochasticHaissinski.jl")

using .StochasticHaissinski
using BenchmarkTools
using Statistics
using LIKWID
using Dates
using Base.Threads
using Printf
using Random
using StructArrays
using MPI
using Profile, ProfileSVG

function parse_command_args()
    parsed_args = Dict{String, Any}()
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--turns" && i < length(ARGS)
            turns_str = ARGS[i+1]
            parsed_args["turns"] = parse(Float64, turns_str)
            i += 2
        elseif arg == "--particles" && i < length(ARGS)
            particles_str = ARGS[i+1]
            parsed_args["particles"] = parse(Float64, particles_str)
            i += 2
        elseif arg == "--mpi"
            # Keep the existing --mpi flag handling
            parsed_args["mpi"] = true
            i += 1
        else
            # Skip unknown arguments
            i += 1
        end
    end
    
    return parsed_args
end

parsed_args = parse_command_args()

# --- Helper Functions for Formatting ---
function format_bytes(bytes)
    bytes = round(Int, bytes) # Ensure integer before comparison
    bytes == 0 && return "0 bytes"
    units = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"]
    val = Float64(bytes)
    idx = 1
    while val >= 1024 && idx < length(units)
        val /= 1024
        idx += 1
    end
    return @sprintf("%.2f %s", val, units[idx])
end

function format_time(t_sec)
    t_sec == 0 && return "0.0 s"
    units = ["s", "ms", "μs", "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    idx = 1
    val = t_sec
    # Find appropriate unit
    while idx < length(units) && (val * scaling[idx+1]) < 1000
        idx += 1
        val = t_sec * scaling[idx]
        # Break if we reached smallest unit or value is reasonable
        if idx == length(units) || val >= 1
             break
        end
    end
     # Final check if we are at seconds but value is small
     if idx == 1 && val < 1
         idx = 2 # Go to ms
         val = t_sec * scaling[idx]
     end

    return @sprintf("%.3f %s", val, units[idx])
end


# --- Determine Run Mode from Command Line ---
# Check if "--mpi" flag is present in the command line arguments
const run_mpi_flag = "--mpi" in ARGS


# --- MPI Initialization (Conditional) ---
comm = nothing
rank = 0
comm_size = 1
if run_mpi_flag
    if !MPI.Initialized()
        # Only initialize MPI if the flag is set and it's not already initialized
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    if rank == 0
        println("MPI mode requested (--mpi flag detected). Running with $comm_size ranks.")
    end
else
    # Ensure message only prints once if not using MPI
    if rank == 0 # This condition is technically always true here, but good practice
        println("Serial mode requested (no --mpi flag). Running on a single process.")
    end
end

# --- Declare Script-Level Variables ---
# Variables needed across different scopes declared here
local particles
local E0::Float64
local n_local::Int
local log_file
local log_output
local benchmark_results # Store BenchmarkTools results locally on rank 0 if needed

# --- Parameters (Identical on all ranks/processes) ---
num_threads = Threads.nthreads() # Julia threads per process/rank
log_dir_base = joinpath(project_root, "logs") # Log dir relative to project root
log_subdir = run_mpi_flag ? "mpi_run" : "serial_run"
log_dir = joinpath(log_dir_base, log_subdir)
parameters = Dict{String, Any}()

# Physics parameters (defined globally)
begin
    E0_ini = 4e9
    mass = MASS_ELECTRON
    voltage = 5e6
    harmonic = 360
    radius = 250.0
    pipe_radius = 0.00025
    α_c = 3.68e-4
    γ = E0_ini/mass
    β = sqrt(1 - 1/γ^2);
    η = α_c - 1/γ^2
    sin_ϕs = 0.5;
    ϕs = 5π/6
    freq_rf = (ϕs + 10*π/180) * β * SPEED_LIGHT / (2π)
end;
# Distribution parameters (defined globally)
begin
    μ_z = 0.0
    μ_E = 0.0
    T_rev = (2*π*radius) / (β*SPEED_LIGHT)
    ω_rev = 2π / T_rev
    σ_E0 = 1e6
    cos_ϕs_val = cos(ϕs)
    σ_z0_factor = α_c*E0_ini/(harmonic*voltage*abs(cos_ϕs_val)) # Use abs for safety if needed
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(σ_z0_factor) * σ_E0 / E0_ini

end;


# --- Simulation Setup ---
# n_turns = Int64(1e2) # Number of turns
# n_particles_global = Int64(1e5); # Total number of particles
n_turns = haskey(parsed_args, "turns") ? Int64(parsed_args["turns"]) : Int64(1e2)
n_particles_global = haskey(parsed_args, "particles") ? Int64(parsed_args["particles"]) : Int64(1e5)

# --- Particle Generation & Distribution ---
if run_mpi_flag
    # --- MPI Mode: Rank 0 generates, all ranks calculate distribution ---
    counts = Vector{Int}(undef, comm_size)
    displs = Vector{Int}(undef, comm_size)
    base_n_local = n_particles_global ÷ comm_size
    remainder = n_particles_global % comm_size
    local current_displacement = 0
    for r in 0:(comm_size-1)
        local_count = r < remainder ? base_n_local + 1 : base_n_local
        counts[r+1] = local_count
        displs[r+1] = current_displacement
        current_displacement += local_count
    end
    n_local = counts[rank+1]

    # Rank 0 generates all particles
    local particles_global_struct # Only defined on Rank 0 temporarily
    if rank == 0
        println("Rank 0 generating $n_particles_global initial global particles...")
        Random.seed!(1234) # Seed RNG for reproducibility
        particles_global_struct, _, _, E0_generated = StochasticHaissinski.generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles_global, E0_ini, mass, ϕs, freq_rf);
        E0 = E0_generated # Assign to outer scope E0
        # Extract coordinate arrays for scattering
        particles_global_z = particles_global_struct.coordinates.z
        particles_global_dE = particles_global_struct.coordinates.ΔE
    else
        # Allocate space for incoming particles
        local_z_coords = Vector{Float64}(undef, n_local)
        local_dE_coords = Vector{Float64}(undef, n_local)
        local_coords = StructArray{Coordinate{Float64}}((z=local_z_coords, ΔE=local_dE_coords))
        # Assign to outer scope particles
        particles = StructArray{Particle{Float64}}((coordinates=local_coords,))
        E0 = 0.0 # Placeholder, will be overwritten by broadcast
    end

    # Broadcast E0 from Rank 0
    ref_E0 = Ref(E0); MPI.Bcast!(ref_E0, 0, comm)
    if rank != 0; E0 = ref_E0[]; end # Update E0 on non-root ranks

    # Scatter Particle Data using Scatterv!
    # Prepare receive buffers
    rbuf_z = MPI.Buffer(rank == 0 ? Vector{Float64}(undef, n_local) : particles.coordinates.z)
    rbuf_dE = MPI.Buffer(rank == 0 ? Vector{Float64}(undef, n_local) : particles.coordinates.ΔE)

    # Prepare send buffers (only on Rank 0)
    sbuf_z = nothing; sbuf_dE = nothing
    if rank == 0
        sbuf_z = MPI.VBuffer(particles_global_z, counts, displs)
        sbuf_dE = MPI.VBuffer(particles_global_dE, counts, displs)
    end

    # Perform Scatterv
    MPI.Scatterv!(sbuf_z, rbuf_z, 0, comm)
    MPI.Scatterv!(sbuf_dE, rbuf_dE, 0, comm)

    # Rank 0: Create its local `particles` StructArray from the received buffer data
    if rank == 0
        z_coords_local = rbuf_z.data # Extract the data vector
        dE_coords_local = rbuf_dE.data
        coords_local = StructArray{Coordinate{Float64}}((z=z_coords_local, ΔE=dE_coords_local))
        # Assign to outer scope particles
        particles = StructArray{Particle{Float64}}((coordinates=coords_local,))
    end

    MPI.Barrier(comm) # Synchronize after scatter
    if rank == 0; println("Scatterv finished. All ranks have local particles."); end

else
    # --- Serial Mode: Generate all particles directly ---
    if rank == 0 println("Generating $n_particles_global initial particles (Serial mode)..."); end
    Random.seed!(1234)
    # Assign to outer scope variables
    particles, _, _, E0 = StochasticHaissinski.generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles_global, E0_ini, mass, ϕs, freq_rf);
    n_local = n_particles_global # In serial, local count is global count
    if rank == 0 println("Finished generating particles (Serial mode)."); end
end

# --- Calculate nbins ---
# Base calculation on global particle count, ensure power of two
nbins_calc = StochasticHaissinski.next_power_of_two(max(64, Int(round(n_particles_global / 100))))

# --- Create Sim Params & Buffers (All Ranks/Processes) ---
# Use the determined E0 (generated or broadcasted)
sim_params = SimulationParameters(E0, mass, voltage, harmonic, radius, pipe_radius, α_c, ϕs, freq_rf, n_turns, true, true, true, true, true);
# Buffers sized based on n_local (MPI) or n_global (Serial)
buffers = StochasticHaissinski.create_simulation_buffers(n_local, nbins_calc, run_mpi_flag, comm_size; T=Float64);

# --- Pre-run / Compilation ---
if rank == 0; println("Performing pre-run for compilation..."); end
pre_params = SimulationParameters( E0, mass, voltage, harmonic, radius, pipe_radius, α_c, ϕs, freq_rf, 1, # Only 1 turn
                                    sim_params.use_wakefield, sim_params.update_η, sim_params.update_E0,
                                    sim_params.SR_damping, sim_params.use_excitation);
# Use deepcopy to avoid modifying original data/buffers during compilation run
particles_copy = deepcopy(particles); buffers_copy = deepcopy(buffers)
# Call evolve with the correct flag
StochasticHaissinski.longitudinal_evolve!(particles_copy, pre_params, buffers_copy, comm, run_mpi_flag);
if run_mpi_flag; MPI.Barrier(comm); end # Sync after compilation run
if rank == 0; println("Compilation run finished."); end

# --- Rank 0: Setup Logging ---
if rank == 0
    mkpath(log_dir) # Create log directory if it doesn't exist
    parameters["E0_ini"] = E0_ini; parameters["E0_start"] = E0; parameters["mass"] = mass
    parameters["voltage"] = voltage; parameters["harmonic"] = harmonic; parameters["radius"] = radius
    parameters["pipe_radius"] = pipe_radius; parameters["α_c"] = α_c; parameters["γ"] = γ
    parameters["β"] = β; parameters["η"] = η; parameters["sin_ϕs"] = sin_ϕs; parameters["ϕs"] = ϕs
    parameters["freq_rf"] = freq_rf; parameters["μ_z"] = μ_z; parameters["μ_E"] = μ_E
    parameters["ω_rev"] = ω_rev; parameters["σ_E0"] = σ_E0; parameters["σ_z0"] = σ_z0
    parameters["n_turns"] = n_turns; parameters["n_particles_global"] = n_particles_global
    parameters["num_threads_per_process"] = num_threads; parameters["mpi_comm_size"] = comm_size
    parameters["nbins_calculated"] = nbins_calc; parameters["run_mode"] = run_mpi_flag ? "MPI" : "Serial"

    # Format filename components
    turns_raw = @sprintf("%.0e", Float64(n_turns)); particles_raw = @sprintf("%.0e", Float64(n_particles_global))
    turns_sci = replace(replace(replace(turns_raw, "." => ""), "e+" => "e"), r"e0+" => "e")
    particles_sci = replace(replace(replace(particles_raw, "." => ""), "e+" => "e"), r"e0+" => "e")
    mode_tag = run_mpi_flag ? "_mpi$(comm_size)" : "_serial"

    log_filename = "turns$(turns_sci)_particles$(particles_sci)$(mode_tag)_threads$(num_threads).log"
    log_path = joinpath(log_dir, log_filename)
    # Use assignment that works inside conditional block
    log_file_ref = Ref{IOStream}() # Use a reference
    try
        log_file_ref[] = open(log_path, "w")
        # Assign to outer scope log_file using global keyword
        global log_file = log_file_ref[]
    catch e
        println("ERROR: Could not open log file '$log_path': $e")
        # Define log_output to print to console as fallback
        # Assign to outer scope log_output using global keyword
         global log_output = (args...) -> println("LOG_FALLBACK: ", args...)
         # Ensure log_file is not assigned if open failed
         global log_file = nothing
    end

    # Define log_output function only if file opened successfully
    if @isdefined(log_file) && log_file !== nothing && isopen(log_file)
        write(log_file, "=== Simulation Log ($(parameters["run_mode"]) Mode) ===\n")
        write(log_file, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n"); write(log_file, "=== Parameters ===\n")
        param_names = ["run_mode", "E0_ini", "E0_start", "mass", "voltage", "harmonic", "radius", "pipe_radius",
                       "α_c", "γ", "β", "η", "sin_ϕs", "ϕs", "freq_rf", "μ_z", "μ_E",
                       "ω_rev", "σ_E0", "σ_z0", "n_turns", "n_particles_global",
                       "num_threads_per_process", "mpi_comm_size", "nbins_calculated"]
        for name in param_names; value = parameters[name]; write(log_file, "$name = $value\n"); end
        write(log_file, "\n=== Simulation Output ===\n")

        # Define log_output function (captures log_file)
        function _log_output_rank0(args...)
            msg = string(args...)
            # Check again inside function for safety
            if @isdefined(log_file) && log_file !== nothing && isopen(log_file)
                 write(log_file, msg, "\n"); flush(log_file);
            else
                  println("Warning (Rank 0): log_file closed or unavailable. Message: ", msg) # Log to console as fallback
            end
        end
        # Assign to outer scope log_output using global keyword
        global log_output = _log_output_rank0
    end

end # End Rank 0 logging setup


# --- Measurement Section ---
if rank == 0 && @isdefined(log_output) log_output("Starting BenchmarkTools run (local) & aggregation..."); end

# Use copies for benchmarking to allow multiple runs without resetting state
particles_bench = deepcopy(particles); buffers_bench = deepcopy(buffers)
bench_sim_params = sim_params # Parameters are immutable

# Define the benchmarkable expression
bench_run = @benchmarkable StochasticHaissinski.longitudinal_evolve!($particles_bench, $bench_sim_params, $buffers_bench, $comm, $run_mpi_flag) setup=(
    # Create fresh copies for this sample evaluation
    particles_bench = deepcopy($particles);
    buffers_bench = deepcopy($buffers);
    # Synchronize before the timing starts for this evaluation
    if $run_mpi_flag; MPI.Barrier($comm); end
) teardown=(
    # Synchronize after the timing ends for this evaluation
    if $run_mpi_flag; MPI.Barrier($comm); end
)

# Run the benchmark - all ranks participate
# Increase samples/evals if needed for stability, but start low
results = BenchmarkTools.run(bench_run, samples=10, evals=1)

# Extract local median results
local_median_time_ns = time(median(results))  # Time in nanoseconds
local_median_bytes = Float64(memory(median(results))) # Bytes as Float64
local_median_gctime_ns = gctime(median(results)) # GC time in nanoseconds
# Manually calculate median allocations
allocs_vector = results.allocs # Get vector of allocs per sample
local_median_allocs = isempty(allocs_vector) ? 0.0 : Float64(median(allocs_vector)) # Calculate median, handle empty case


# --- Aggregate Median Results ---
# Max Median Time (proxy for Wall Time)
# Convert local time to seconds for consistency before reducing
local_median_time_s = local_median_time_ns * 1e-9
global_max_median_time = run_mpi_flag ? MPI.Reduce(local_median_time_s, MPI.MAX, 0, comm) : local_median_time_s

# Sum Median Bytes, GC Time, and Allocations
local_median_gctime_s = local_median_gctime_ns * 1e-9 # Convert GC time to seconds
if run_mpi_flag
    # Add local_median_allocs to the reduction
    total_stats_reduced = MPI.Reduce([local_median_bytes, local_median_gctime_s, local_median_allocs], MPI.SUM, 0, comm)
    # Assign results only on rank 0
    total_median_bytes = (rank == 0) ? total_stats_reduced[1] : 0.0
    total_median_gctime = (rank == 0) ? total_stats_reduced[2] : 0.0
    total_median_allocs = (rank == 0) ? Int(round(total_stats_reduced[3])) : 0 # Convert back to Int
else # Serial case
    total_median_bytes = local_median_bytes
    total_median_gctime = local_median_gctime_s
    total_median_allocs = Int(round(local_median_allocs)) # Convert back to Int
end

# --- Rank 0 Logging of Aggregated Benchmark Results ---
if rank == 0 && @isdefined(log_output)
    # Calculate GC time percentage using aggregated values
    gc_time_percentage = (global_max_median_time > 0) ? (total_median_gctime / global_max_median_time * 100) : 0.0

    log_output("\nAggregated Benchmark Results (Based on Local Medians):")
    # Use formatting helpers
    log_output("  Max Median Time (across ranks): $(format_time(global_max_median_time))")
    log_output("  Sum of Median Memory Allocated: $(format_bytes(total_median_bytes))")
    log_output("  Sum of Median GC Time: $(format_time(total_median_gctime)) ($(Printf.@sprintf("%.2f", gc_time_percentage))%)")
    log_output(@sprintf("  Sum of Median Allocations: %.5g", total_median_allocs))
    log_output("-"^20)
end


# --- Performance Monitoring Section (LIKWID - Single Run, Output Suppressed) ---
if rank == 0 && @isdefined(log_output) log_output("\nStarting LIKWID performance monitoring ('FLOPS_DP')..."); end

# Use fresh copies for the perfmon run
particles_perf = deepcopy(particles); buffers_perf = deepcopy(buffers)
perf_sim_params = sim_params

# Declare the variable to hold this rank's result *before* the try/catch
local local_flop_count::Float64 = NaN # Initialize before try/catch
original_stdout = stdout   

try
    if run_mpi_flag; MPI.Barrier(comm); end # Sync before perfmon start

    redirect_stdout(devnull) # <<< Redirect stdout HERE

    # Run the code block under LIKWID monitoring
    _, events = @perfmon "FLOPS_DP" begin
        StochasticHaissinski.longitudinal_evolve!(particles_perf, perf_sim_params, buffers_perf, comm, run_mpi_flag)
    end


    redirect_stdout(original_stdout)

    if run_mpi_flag; MPI.Barrier(comm); end # Sync after perfmon end

    # Extract FLOP count if available
    if haskey(events, "FLOPS_DP") && isa(events["FLOPS_DP"], Vector) && !isempty(events["FLOPS_DP"]) && haskey(events["FLOPS_DP"][1], "RETIRED_SSE_AVX_FLOPS_ALL")
        # Assign to the local variable declared outside the try block using global
        global local_flop_count = Float64(events["FLOPS_DP"][1]["RETIRED_SSE_AVX_FLOPS_ALL"])
    else
        # Print message to original stdout if LIKWID data is missing
        println(original_stdout, "Rank $rank: LIKWID 'FLOPS_DP' counter key not found or data structure unexpected.");
        global local_flop_count = NaN # Ensure it's NaN if extraction failed
    end
catch e
    # <<< Ensure stdout restored even on error BEFORE printing error message
    redirect_stdout(original_stdout)
    # Print message to original stdout if LIKWID fails
    println(original_stdout, "Rank $rank: LIKWID performance monitoring failed: ", e);
    global local_flop_count = NaN # Ensure it's NaN if try block failed
    # Ensure barrier is still called in case of error on some ranks
    if run_mpi_flag; MPI.Barrier(comm); end
finally
    # <<< Ensure stdout is *always* restored
    if stdout != original_stdout
        redirect_stdout(original_stdout)
    end
end

# Gather FLOP counts (NaN if failed/skipped) to Rank 0
send_buf = Ref(local_flop_count); recv_buf = nothing
if run_mpi_flag
    if rank == 0; recv_buf = Vector{Float64}(undef, comm_size); end
    MPI.Gather!(send_buf, recv_buf, 0, comm)
else
    # In serial mode, recv_buf is just the local count
    if rank == 0; recv_buf = [local_flop_count]; end
end


# Rank 0: Process and Log LIKWID Results
if rank == 0 && @isdefined(log_output)
    log_output("\nPerformance monitoring results (Aggregated LIKWID FLOPS_DP):")
    # Explicitly declare these variables as local to this block
    local total_flops = 0.0
    local any_rank_failed = false
    local valid_flops_collected = 0

    if recv_buf !== nothing
        current_comm_size = run_mpi_flag ? comm_size : 1 # Get actual size for loop
        for r_idx in 1:current_comm_size
            r_rank = r_idx - 1 # 0-based rank index
            flops_r = recv_buf[r_idx]
            if isnan(flops_r)
                log_output("Rank $r_rank: FLOP count collection failed or unavailable.")
                any_rank_failed = true # Assignment to local variable
            else
                log_output("Rank $r_rank: Local FLOPs = $(@sprintf("%.3e", flops_r))")
                # Check for non-NaN before adding
                if !isnan(flops_r)
                    total_flops += flops_r             
                    valid_flops_collected += 1       
                else 
                    any_rank_failed = true             
                end
            end
        end

        if any_rank_failed
            log_output("Total FLOPs (approximate, from $valid_flops_collected ranks): $(@sprintf("%.3e", total_flops))")
        elseif valid_flops_collected > 0
            log_output("Total aggregated FLOPs: $(@sprintf("%.3e", total_flops))")
        else
                log_output("No valid FLOP counts collected from any rank.")
        end

        # Calculate GFLOPS rate using the Max Median Time from BenchmarkTools
        # Use global_max_median_time calculated earlier
        # Check variables are defined and valid before calculation
        if @isdefined(global_max_median_time) && !any_rank_failed && total_flops > 0 && global_max_median_time > 0 && valid_flops_collected == current_comm_size
             gflops_rate = total_flops / global_max_median_time / 1e9
             log_output("Aggregated GFLOPS rate (based on Max Median Time): $(@sprintf("%.3f", gflops_rate)) GFLOPS")
             # Calculate FLOPs per particle per turn (using global counts)
             flops_per_particle_per_turn = total_flops / n_particles_global / n_turns
             log_output("FLOPs per particle per turn: $(@sprintf("%.3f", flops_per_particle_per_turn))")
        else
            log_output("GFLOPS rate calculation skipped (FLOP count failed, zero FLOPs, zero max median time, or benchmark results missing).")
        end
    else
        log_output("Failed to receive FLOP counts from ranks.")
    end
    log_output("-"^20)
end # End Rank 0 LIKWID processing


# --- Final Cleanup ---
if rank == 0
    # Close the log file if it was successfully opened
    if @isdefined(log_file) && log_file !== nothing && isopen(log_file)
        # Check log_output exists before using it
        if @isdefined(log_output) log_output("\nSimulation complete.") end
        close(log_file)
        println("Simulation complete. Results logged.")
    elseif @isdefined(log_output) # Log file failed to open
            log_output("\nSimulation complete (log file could not be opened).")
            println("Simulation complete (log file could not be opened).")
    else # Should not happen if Rank 0 logic is correct
            println("Simulation complete (logging state unclear).")
    end
end

# if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    # @profile begin
    #     # Run the function ONCE to collect profile data
    #     StochasticHaissinski.longitudinal_evolve!(particles_perf, perf_sim_params, buffers_perf, comm, run_mpi_flag; show_progress=false)
    # end
    # println("Rank 0: Profile data collected. Saving SVG...")
    # ProfileSVG.save("prof.svg")
#     Profile.Allocs.@profile sample_rate=0.01 StochasticHaissinski.longitudinal_evolve!(particles_perf, perf_sim_params, buffers_perf, comm, run_mpi_flag; show_progress=false)
#     results = Profile.Allocs.fetch()
#     allocs_sorted = sort(results.allocs, by=x->x.size, rev=true) # Sort descending

#     num_to_show = 10 # How many of the top allocations to display
#     println("\nTop $(min(num_to_show, length(allocs_sorted))) allocations by size:")
#     for i in 1:min(num_to_show, length(allocs_sorted))
#         alloc = allocs_sorted[i]
#         # Print type, size, and maybe the first few frames of the stack trace
#         println("  #$i: Size=$(alloc.size), Type=$(alloc.type)")
#         # Optional: Print top of stack trace (can be long)
#         println("      Stacktrace (top):")
#         for j in 1:min(25, length(alloc.stacktrace)) # Show top 5 frames
#             println("        - $(alloc.stacktrace[j])")
#         end
#         if length(alloc.stacktrace) > 5
#             println("        - ... (and $(length(alloc.stacktrace)-5) more)")
#         end
#     end
# end



# Finalize MPI 
if run_mpi_flag && MPI.Initialized() && !MPI.Finalized()
    # Add a barrier before finalizing to ensure all ranks are ready
    MPI.Barrier(comm)
    MPI.Finalize()
end

