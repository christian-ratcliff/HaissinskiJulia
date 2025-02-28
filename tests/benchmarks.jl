include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using BenchmarkTools
    using Statistics
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
        100,       # n_turns
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
    σ_E_final, σ_z_final, E0_final = longitudinal_evolve!(particles, sim_params, buffers)
    particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
end;

@btime longitudinal_evolve!(particles, sim_params, buffers)
# 191.733 ms (5733 allocations: 189.86 MiB)  1e5 particles, 1e2 turns, less allcoaations, slightly longer, much more memory

#current: 209.533 ms (6990 allocations: 189.88 MiB)  1e5 particles, 1e2 turns
@benchmark longitudinal_evolve!(particles, sim_params, buffers)
# BenchmarkTools.Trial: 26 samples with 1 evaluation.
#  Range (min … max):  187.889 ms … 203.816 ms  ┊ GC (min … max): 2.80% … 3.04%
#  Time  (median):     193.475 ms               ┊ GC (median):    2.85%
#  Time  (mean ± σ):   193.770 ms ±   3.625 ms  ┊ GC (mean ± σ):  3.01% ± 0.40%
#Need to fix this