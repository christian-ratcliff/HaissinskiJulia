include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using BenchmarkTools
    using Statistics
    using LIKWID
end

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
    n_turns = 100
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
        n_turns,       # n_turns
        true,        # use_wakefield
        true,        # update_η
        true,        # update_E0
        true,        # SR_damping
        true         # use_excitation
    );
    # Generate particles
    n_particles = Int64(1e5);
    particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
    buffers = create_simulation_buffers(n_particles, Int(n_particles/10), Float64);
    # Run simulation
    σ_E_final, σ_z_final, E0_final = longitudinal_evolve!(particles, sim_params, buffers; show_progress=false);
    particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
end;

@btime longitudinal_evolve!(particles, sim_params, buffers; show_progress=false)
#  128.272 ms (7937 allocations: 43.20 MiB)  1e5 particles, 1e2 turns, less allcoaations, slightly longer, much more memory

#current: 209.533 ms (6990 allocations: 189.88 MiB)  1e5 particles, 1e2 turns
benchmark_results = @benchmark longitudinal_evolve!(particles, sim_params, buffers; show_progress=false)
display(benchmark_results)
# BenchmarkTools.Trial: 34 samples with 1 evaluation per sample.
#  Range (min … max):  130.563 ms … 199.680 ms  ┊ GC (min … max): 0.60% … 0.81%
#  Time  (median):     143.747 ms               ┊ GC (median):    0.43%
#  Time  (mean ± σ):   147.825 ms ±  18.484 ms  ┊ GC (mean ± σ):  0.44% ± 0.12%
#  Memory estimate: 43.20 MiB, allocs estimate: 7937.

metrics, events = @perfmon "FLOPS_DP" begin
    longitudinal_evolve!(particles, sim_params, buffers; show_progress=false)
end

# Display FLOP count
flop_count = first(events["FLOPS_DP"])["RETIRED_SSE_AVX_FLOPS_ALL"]
println("Total FLOPs: ", flop_count)

# Calculate GFLOPS rate
execution_time = median(benchmark_results.times) / 1e9  # Convert ns to seconds
gflops_rate = flop_count / execution_time / 1e9
println("GFLOPS rate: ", gflops_rate)

# You can also calculate FLOPs per particle per turn
flops_per_particle_per_turn = flop_count / n_particles / n_turns
println("FLOPs per particle per turn: ", flops_per_particle_per_turn)