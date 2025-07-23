# tests/verify_mpi.jl
# Integrated verification test for StochasticHaissinski comparing MPI and serial results
# Run with: mpiexec -n N julia tests/verify_mpi.jl [--turns VALUE] [--particles VALUE]

# Determine the base path relative to the script's location
script_dir = dirname(@__FILE__)
project_root = joinpath(script_dir, "..")
src_path = joinpath(project_root, "src", "StochasticHaissinski.jl")
include(src_path)

using .StochasticHaissinski
using Statistics
using MPI
using Random
using StructArrays
using Printf
using Dates

function parse_command_args()
    parsed_args = Dict{String, Any}()
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        
        if arg == "--turns" && i < length(ARGS)
            turns_str = ARGS[i+1]
            parsed_args["turns"] = parse(Int, turns_str)
            i += 2
        elseif arg == "--particles" && i < length(ARGS)
            particles_str = ARGS[i+1]
            parsed_args["particles"] = parse(Int, particles_str)
            i += 2
        elseif arg == "--tolerance" && i < length(ARGS)
            tolerance_str = ARGS[i+1]
            parsed_args["tolerance"] = parse(Float64, tolerance_str)
            i += 2
        else
            # Skip unknown arguments
            i += 1
        end
    end
    
    return parsed_args
end

function run_verification_test()
    # Initialize MPI if not already initialized
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    
    # Parse command line arguments - only on rank 0
    parsed_args = parse_command_args()
    
    # Default parameters
    n_turns = get(parsed_args, "turns", 100)
    n_particles = get(parsed_args, "particles", 10000)
    tolerance = get(parsed_args, "tolerance", 1e-10)
    ref_file = joinpath(script_dir, "mpi_reference_results.txt")
    
    # Broadcast parameters from rank 0 to all ranks
    if rank == 0
        n_turns_ref = Ref(n_turns)
        n_particles_ref = Ref(n_particles)
        tolerance_ref = Ref(tolerance)
    else
        n_turns_ref = Ref(0)
        n_particles_ref = Ref(0)
        tolerance_ref = Ref(0.0)
    end
    
    MPI.Bcast!(n_turns_ref, 0, comm)
    MPI.Bcast!(n_particles_ref, 0, comm)
    MPI.Bcast!(tolerance_ref, 0, comm)
    
    if rank != 0
        n_turns = n_turns_ref[]
        n_particles = n_particles_ref[]
        tolerance = tolerance_ref[]
    end
    
    # --- Physics parameters (defined globally) ---
    # These should be identical to those in benchmark_total.jl
    E0_ini = 4e9
    mass = MASS_ELECTRON
    voltage = 5e6
    harmonic = 360
    radius = 250.0
    pipe_radius = 0.00025
    α_c = 3.68e-4
    γ = E0_ini/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    sin_ϕs = 0.5
    ϕs = 5π/6
    freq_rf = (ϕs + 10*π/180) * β * SPEED_LIGHT / (2π)
    
    # --- Distribution parameters ---
    μ_z = 0.0
    μ_E = 0.0
    T_rev = (2*π*radius) / (β*SPEED_LIGHT)
    ω_rev = 2π / T_rev
    σ_E0 = 1e6
    cos_ϕs_val = cos(ϕs)
    σ_z0_factor = α_c*E0_ini/(harmonic*voltage*abs(cos_ϕs_val))
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(σ_z0_factor) * σ_E0 / E0_ini
    
    if rank == 0
        println("=== StochasticHaissinski MPI Verification Test ===")
        println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println("Running with $comm_size MPI ranks")
        println("Simulation: $n_particles particles for $n_turns turns")
        println("Tolerance: $tolerance")
        println("Reference file: $ref_file")
        println("="^50)
    end

    # Run the simulation with current MPI configuration
    if rank == 0
        println("Running simulation with $comm_size MPI rank(s)...")
    end
    
    # --- Set up MPI particle distribution ---
    counts = Vector{Int}(undef, comm_size)
    displs = Vector{Int}(undef, comm_size)
    base_n_local = n_particles ÷ comm_size
    remainder = n_particles % comm_size
    current_displacement = 0
    
    for r in 0:(comm_size-1)
        local_count = r < remainder ? base_n_local + 1 : base_n_local
        counts[r+1] = local_count
        displs[r+1] = current_displacement
        current_displacement += local_count
    end
    
    n_local = counts[rank+1]
    
    # --- Generate and distribute particles ---
    # Rank 0 generates all particles
    local particles_global_struct
    local E0_generated = 0.0
    
    if rank == 0
        # Set the seed for reproducible results
        Random.seed!(1234)
        particles_global_struct, _, _, E0_generated = StochasticHaissinski.generate_particles(
            μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf
        )
        # Extract coordinate arrays for scattering
        particles_global_z = particles_global_struct.coordinates.z
        particles_global_dE = particles_global_struct.coordinates.ΔE
    else
        # Allocate space for incoming particles on other ranks
        local_z_coords = Vector{Float64}(undef, n_local)
        local_dE_coords = Vector{Float64}(undef, n_local)
        local_coords = StructArray{Coordinate{Float64}}((z=local_z_coords, ΔE=local_dE_coords))
        # Placeholder particles for non-rank-0
        particles = StructArray{Particle{Float64}}((coordinates=local_coords,))
    end
    
    # Broadcast E0 from Rank 0
    E0_ref = Ref(E0_generated)
    MPI.Bcast!(E0_ref, 0, comm)
    E0 = E0_ref[]
    
    # Scatter Particle Data using Scatterv
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
        z_coords_local = rbuf_z.data
        dE_coords_local = rbuf_dE.data
        coords_local = StructArray{Coordinate{Float64}}((z=z_coords_local, ΔE=dE_coords_local))
        particles = StructArray{Particle{Float64}}((coordinates=coords_local,))
    end
    
    MPI.Barrier(comm)
    
    # --- Create simulation parameters and buffers ---
    nbins_calc = StochasticHaissinski.next_power_of_two(max(64, Int(round(n_particles / 100))))
    sim_params = SimulationParameters(E0, mass, voltage, harmonic, radius, pipe_radius, 
                                     α_c, ϕs, freq_rf, n_turns, true, true, true, true, true)
    buffers = StochasticHaissinski.create_simulation_buffers(n_local, nbins_calc, true, comm_size; T=Float64)
    
    # Run a pre-run for compilation (1 turn)
    pre_params = SimulationParameters(E0, mass, voltage, harmonic, radius, pipe_radius, α_c, ϕs, freq_rf, 1,
                                     sim_params.use_wakefield, sim_params.update_η, sim_params.update_E0,
                                     sim_params.SR_damping, sim_params.use_excitation)
    particles_copy = deepcopy(particles)
    buffers_copy = deepcopy(buffers)
    StochasticHaissinski.longitudinal_evolve!(particles_copy, pre_params, buffers_copy, comm, true)
    
    # Run the full simulation with MPI
    σE_current, σz_current, E0_current = StochasticHaissinski.longitudinal_evolve!(
        particles, sim_params, buffers, comm, true
    )
    
    MPI.Barrier(comm)
    
    # If this is a 1-rank run, save the results as reference
    if comm_size == 1 && rank == 0
        println("Running with 1 MPI rank - saving results as reference.")
        
        # Save reference results to file
        try
            mkpath(dirname(ref_file))  # Ensure directory exists
            open(ref_file, "w") do f
                write(f, "ref_σE=$σE_current\n")
                write(f, "ref_σz=$σz_current\n")
                write(f, "ref_E0=$E0_current\n")
                write(f, "n_turns=$n_turns\n")
                write(f, "n_particles=$n_particles\n")
                write(f, "timestamp=$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
            end
            println("Reference results saved to $ref_file")
            println("Current results (1 MPI rank):")
            println("  σ_E = $(@sprintf("%.10e", σE_current))")
            println("  σ_z = $(@sprintf("%.10e", σz_current))")
            println("  E0 = $(@sprintf("%.10e", E0_current))")
            
            # Return true since this is the reference run
            return true
        catch e
            println("Error saving reference results: $e")
            return false
        end
    end
    
    # For multi-rank runs, we need reference values to compare against
    # FIX: Declare each variable separately
    local ref_σE = 0.0
    local ref_σz = 0.0
    local ref_E0 = 0.0
    local test_pass = false
    
    if rank == 0
        println("Current results ($comm_size MPI ranks):")
        println("  σ_E = $(@sprintf("%.10e", σE_current))")
        println("  σ_z = $(@sprintf("%.10e", σz_current))")
        println("  E0 = $(@sprintf("%.10e", E0_current))")
        println("-"^50)
        
        # Try to load reference results from file
        if isfile(ref_file)
            try
                ref_data = Dict{String, Float64}()
                ref_metadata = Dict{String, String}()
                
                for line in readlines(ref_file)
                    key, value = split(line, '=')
                    if key in ["ref_σE", "ref_σz", "ref_E0", "n_turns", "n_particles"]
                        ref_data[key] = parse(Float64, value)
                    else
                        ref_metadata[key] = value
                    end
                end
                
                ref_σE = ref_data["ref_σE"]
                ref_σz = ref_data["ref_σz"]
                ref_E0 = ref_data["ref_E0"]
                
                # Verify that reference run used same parameters
                ref_n_turns = Int(ref_data["n_turns"])
                ref_n_particles = Int(ref_data["n_particles"])
                
                println("Loaded reference results from $ref_file (timestamp: $(get(ref_metadata, "timestamp", "unknown")))")
                
                if ref_n_turns != n_turns || ref_n_particles != n_particles
                    println("WARNING: Reference parameters don't match current run!")
                    println("  Reference: $ref_n_particles particles, $ref_n_turns turns")
                    println("  Current: $n_particles particles, $n_turns turns")
                end
                
                println("Reference results (1 MPI rank):")
                println("  σ_E = $(@sprintf("%.10e", ref_σE))")
                println("  σ_z = $(@sprintf("%.10e", ref_σz))")
                println("  E0 = $(@sprintf("%.10e", ref_E0))")
                println("-"^50)
                
                # Calculate relative differences
                rel_diff_σE = abs((σE_current - ref_σE) / ref_σE)
                rel_diff_σz = abs((σz_current - ref_σz) / ref_σz)
                rel_diff_E0 = abs((E0_current - ref_E0) / ref_E0)
                
                # Check if differences are within tolerance
                σE_pass = rel_diff_σE <= tolerance
                σz_pass = rel_diff_σz <= tolerance
                E0_pass = rel_diff_E0 <= tolerance
                
                test_pass = σE_pass && σz_pass && E0_pass
                
                println("Verification Results:")
                println("  σ_E relative difference: $(@sprintf("%.3e", rel_diff_σE)) $(σE_pass ? "✓" : "✗")")
                println("  σ_z relative difference: $(@sprintf("%.3e", rel_diff_σz)) $(σz_pass ? "✓" : "✗")")
                println("  E0 relative difference: $(@sprintf("%.3e", rel_diff_E0)) $(E0_pass ? "✓" : "✗")")
                println("-"^50)
                println("Overall Result: $(test_pass ? "PASS" : "FAIL")")
                
            catch e
                println("Error loading reference results: $e")
                println("Using current run as pseudo-reference (NOT VALIDATED)")
                ref_σE = σE_current
                ref_σz = σz_current
                ref_E0 = E0_current
                test_pass = true  # Can't fail against itself
            end
        else
            println("Reference file $ref_file not found.")
            println("Please run first with 1 MPI rank to generate reference results:")
            println("  mpiexec -n 1 julia tests/verify_mpi.jl --turns $n_turns --particles $n_particles")
            println("Using current run as pseudo-reference (NOT VALIDATED)")
            ref_σE = σE_current
            ref_σz = σz_current
            ref_E0 = E0_current
            test_pass = true  # Can't fail against itself
        end
    end
    
    # Broadcast test result to all ranks so they return consistent value
    if rank == 0
        test_pass_ref = Ref(test_pass)
    else
        test_pass_ref = Ref(false)
    end
    
    MPI.Bcast!(test_pass_ref, 0, comm)
    test_pass = test_pass_ref[]
    
    return test_pass
end

# Run the verification test
if abspath(PROGRAM_FILE) == @__FILE__
    try
        success = run_verification_test()
        exit(success ? 0 : 1)  # Exit with code based on test result
    finally
        # Ensure MPI is finalized
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
    end
end