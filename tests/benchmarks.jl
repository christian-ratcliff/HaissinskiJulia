include("../src/StochasticHaissinski.jl")


begin
    using .StochasticHaissinski
    using BenchmarkTools
    using Statistics
    using LIKWID
    using Dates
    using Base.Threads
    using Printf

    num_threads = Threads.nthreads()
end
log_dir = joinpath(dirname(@__FILE__), "..", "logs/initial")
mkpath(log_dir)
parameters = Dict{String, Any}()

begin
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
end;

# Distribution parameters
begin
    μ_z = 0.0
    μ_E = 0.0
    ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT))
    σ_E0 = 1e6
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini
end;


begin
    n_turns = Int64(1e3)
    # Create simulation parameters
    sim_params = SimulationParameters(
        E0_ini,      # E0
        mass,        # mass
        voltage,     # voltage
        harmonic,    # harmonic
        radius,      # radius
        pipe_radius, # pipe_radius
        α_c,         # α_c
        ϕs,          # ϕs
        freq_rf,     # freq_rf
        n_turns,     # n_turns
        true,        # use_wakefield
        true,        # update_η
        true,        # update_E0
        true,        # SR_damping
        true         # use_excitation
    );
    # Generate particles
    n_particles = Int64(1e5);
    particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
    buffers = create_simulation_buffers(n_particles, Int(n_particles/100), Float64);
    # Run simulation
    σ_E_final, σ_z_final, E0_final = longitudinal_evolve!(particles, sim_params, buffers; show_progress=false);
    particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
end;

begin
    # Capture all parameters
    parameters["E0_ini"] = E0_ini
    parameters["mass"] = mass
    parameters["voltage"] = voltage
    parameters["harmonic"] = harmonic
    parameters["radius"] = radius
    parameters["pipe_radius"] = pipe_radius
    parameters["α_c"] = α_c
    parameters["γ"] = γ
    parameters["β"] = β
    parameters["η"] = η
    parameters["sin_ϕs"] = sin_ϕs
    parameters["ϕs"] = ϕs
    parameters["freq_rf"] = freq_rf
    parameters["μ_z"] = μ_z
    parameters["μ_E"] = μ_E
    parameters["ω_rev"] = ω_rev
    parameters["σ_E0"] = σ_E0
    parameters["σ_z0"] = σ_z0
    parameters["n_turns"] = n_turns
    parameters["n_particles"] = n_particles
    parameters["num_threads"] = num_threads
    
    turns_raw = @sprintf("%.0e", Float64(n_turns))
    particles_raw = @sprintf("%.0e", Float64(n_particles))
    
    # Remove decimal point, plus sign, and leading zeros in exponent with multiple replacements
    turns_sci = turns_raw
    turns_sci = replace(turns_sci, "." => "")  # Remove decimal point
    turns_sci = replace(turns_sci, "e+" => "e") # Remove plus sign
    turns_sci = replace(turns_sci, r"e0+" => "e") # Remove leading zeros in exponent
    
    particles_sci = particles_raw
    particles_sci = replace(particles_sci, "." => "")  # Remove decimal point
    particles_sci = replace(particles_sci, "e+" => "e") # Remove plus sign
    particles_sci = replace(particles_sci, r"e0+" => "e") # Remove leading zeros in exponent
    
    # Create log filename with scientific notation
    log_filename = "turns$(turns_sci)_particles$(particles_sci)_threads$(num_threads).log"
    log_path = joinpath(log_dir, log_filename)
    
    # Open log file
    log_file = open(log_path, "w")
    
    # Write header and parameters to log file
    write(log_file, "=== Simulation Log ===\n")
    write(log_file, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
    write(log_file, "=== Parameters ===\n")
    
    # Write all parameters to the log file
    param_names = ["E0_ini", "mass", "voltage", "harmonic", "radius", "pipe_radius", 
                  "α_c", "γ", "β", "η", "sin_ϕs", "ϕs", "freq_rf", "μ_z", "μ_E", 
                  "ω_rev", "σ_E0", "σ_z0", "n_turns", "n_particles", "num_threads"]
    
    for name in param_names
        value = eval(Symbol(name))
        write(log_file, "$name = $value\n")
    end
    
    write(log_file, "\n=== Simulation Output ===\n")
    
    # Function to log output to file only (not terminal)
    function log_output(args...)
        msg = string(args...)
        write(log_file, msg, "\n")
        flush(log_file)
    end
end



# @btime longitudinal_evolve!($particles, $sim_params, $buffers; $show_progress=false)
benchmark_results = @benchmark longitudinal_evolve!($particles, $sim_params, $buffers; show_progress=false)
filtered_results = median(benchmark_results)
io = IOBuffer()
show(io, MIME("text/plain"), filtered_results)
benchmark_details = String(take!(io))
log_output("Benchmark results (median):\n", benchmark_details)




# metrics, events = @perfmon "FLOPS_DP" begin
#     longitudinal_evolve!(particles, sim_params, buffers; show_progress=false)
# end ;
# Create a temporary file for capturing the LIKWID output
temp_file = tempname()
open(temp_file, "w") do temp_io
    # Redirect stdout to the temporary file
    old_stdout = stdout
    redirect_stdout(temp_io)
    
    # Run the performance monitoring
    global metrics, events = @perfmon "FLOPS_DP" begin
        longitudinal_evolve!(particles, sim_params, buffers; show_progress=false)
    end
    
    # Restore stdout
    redirect_stdout(old_stdout)
end

# Read the captured output
perfmon_output = read(temp_file, String)
log_output("Performance monitoring results:\n", perfmon_output)

# Clean up the temporary file
rm(temp_file)

# Display FLOP count
flop_count = first(events["FLOPS_DP"])["RETIRED_SSE_AVX_FLOPS_ALL"]
log_output("Total FLOPs: ", flop_count)

# Calculate GFLOPS rate
execution_time = median(benchmark_results.times) / 1e9  # Convert ns to seconds
gflops_rate = flop_count / execution_time / 1e9
log_output("GFLOPS rate: ", gflops_rate)

# You can also calculate FLOPs per particle per turn
flops_per_particle_per_turn = flop_count / n_particles / n_turns
log_output("FLOPs per particle per turn: ", flops_per_particle_per_turn)

# Close the log file
close(log_file)

# Print just a simple completion message to the terminal
println("Simulation complete. Results logged to: ", log_path)
