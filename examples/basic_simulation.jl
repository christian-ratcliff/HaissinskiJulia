"""
basic_simulation.jl - Basic Haissinski simulation example

This example demonstrates how to:
1. Set up simulation parameters
2. Generate an initial particle distribution
3. Run the evolution simulation
4. Visualize the results
"""

include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using Plots
    using Statistics
end

# # Physical constants
# const SPEED_LIGHT = 299792458.0
# const ELECTRON_CHARGE = 1.602176634e-19
# const MASS_ELECTRON = 0.51099895069e6

# Set physical parameters
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
println("Initial beam parameters: σ_E = $σ_E0 eV, σ_z = $σ_z0 m")

# Create buffers
buffers = create_simulation_buffers(n_particles, Int(n_particles/10), Float64);

# Run simulation
σ_E_final, σ_z_final, E0_final = longitudinal_evolve!(particles, sim_params, buffers)
println("Final beam parameters: σ_E = $σ_E_final eV, σ_z = $σ_z_final m, E0 = $E0_final eV")
# Extract final distribution data
z_values = particles.coordinates.z;
ΔE_values = particles.coordinates.ΔE;

# Create phase space plot
p1 = scatter(z_values ./ σ_z_final, ΔE_values ./ σ_E_final, 
            markersize=1, 
            markerstrokewidth=0, 
            alpha=0.5,
            xlabel="z/σ_z", 
            ylabel="ΔE/σ_E",
            title="Phase Space Distribution",
            label=nothing)

# Create energy distribution histogram
p2 = histogram(ΔE_values ./ σ_E_final, 
              bins=100, 
              normalize=:pdf,
              xlabel="ΔE/σ_E", 
              ylabel="Probability Density",
              title="Energy Distribution",
              label=nothing)

# Create position distribution histogram
p3 = histogram(z_values ./ σ_z_final, 
              bins=100, 
              normalize=:pdf,
              xlabel="z/σ_z", 
              ylabel="Probability Density",
              title="Position Distribution",
              label=nothing)

# Combine plots
plot(p1, p2, p3, layout=(2, 2), size=(1000, 800))
savefig("beam_distribution.png")