include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using Plots
    using Statistics
    using StochasticAD
    using Random
end;

# Set a deterministic seed for reproducibility
Random.seed!(123);

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

    # Distribution parameters
    μ_z = 0.0
    μ_E = 0.0
    ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT))
    σ_E0 = 1e6
    σ_z0 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini

    # Create base simulation parameters with INCREASED number of turns
    base_params = SimulationParameters(
        E0_ini,      # E0
        mass,        # mass
        voltage,     # voltage
        harmonic,    # harmonic
        radius,      # radius
        pipe_radius, # pipe_radius
        α_c,         # α_c
        ϕs,          # ϕs
        freq_rf,     # freq_rf
        500,        # n_turns (increased for better sensitivity detection)
        true,        # use_wakefield
        true,        # update_η
        true,        # update_E0
        true,        # SR_damping
        true         # use_excitation
    )
end;

# Test function to verify StochasticAD is working
function test_stochastic_ad()
    println("Testing StochasticAD with a simple function...")
    
    # Define a simple test function
    f(x) = x^2 + 3*x
    
    # True derivative is 2x + 3
    x_test = 5.0
    true_derivative = 2*x_test + 3
    
    # Compute using StochasticAD
    samples = [derivative_estimate(f, x_test) for _ in 1:100]
    estimated = mean(samples)
    uncertainty = std(samples) / sqrt(100)
    
    println("StochasticAD test at x=$x_test:")
    println("True derivative: $true_derivative")
    println("Estimated: $estimated ± $uncertainty")
    println("Relative error: $(abs(estimated - true_derivative)/true_derivative * 100)%")
    
    return abs(estimated - true_derivative) <= uncertainty * 3
end

# Verify StochasticAD is working
test_result = test_stochastic_ad()
if !test_result
    println("WARNING: StochasticAD test failed. Results may not be reliable.")
else
    println("StochasticAD test passed. Proceeding with sensitivity analysis.")
end

# Generate particles - use fewer particles for faster simulation
n_particles = Int64(1e4)
particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
println("Initial beam parameters: σ_E = $σ_E eV, σ_z = $σ_z m")

# Create parameter transformation and figures of merit
voltage_transform = VoltageTransform()
energy_spread_fom = EnergySpreadFoM()
bunch_length_fom = BunchLengthFoM()


voltage_range = range(3.0e6, 7.0e6, length=3);

# Test a single sensitivity calculation first
println("\nTesting a single sensitivity calculation...")
test_sensitivity = compute_sensitivity(
    voltage_transform,
    energy_spread_fom,
    voltage,  # Use nominal voltage
    particles,
    base_params,
    n_samples=5
)
println("Test sensitivity: $(test_sensitivity.mean_derivative) ± $(test_sensitivity.uncertainty)")

# Scan RF voltage effect on energy spread
println("\nScanning RF voltage effect on energy spread...")
params_energy, foms_energy, grads_energy, errors_energy = scan_parameter(
    voltage_transform,
    energy_spread_fom,
    voltage_range,
    particles,
    base_params,
    n_samples=10  # Use more samples for better statistics
)

# Scan RF voltage effect on bunch length
println("\nScanning RF voltage effect on bunch length...")
params_length, foms_length, grads_length, errors_length = scan_parameter(
    voltage_transform,
    bunch_length_fom,
    voltage_range,
    particles,
    base_params,
    n_samples=10
)

# Plot results


# Energy spread sensitivity plot
p1 = plot_sensitivity_scan(
    params_energy ./ 1e6, 
    foms_energy ./ 1e6, 
    grads_energy ./ 1e6 .* 1e6, # Scale gradient to show dσ_E[MeV]/dV[MV]
    errors_energy ./ 1e6 .* 1e6,
    param_name="RF Voltage [MV]",
    fom_name="Energy Spread [MeV]"
)
savefig(p1, "rf_voltage_energy_spread.png")

# Bunch length sensitivity plot
p2 = plot_sensitivity_scan(
    params_length ./ 1e6, 
    foms_length .* 1e3, # Convert to mm
    grads_length .* 1e3 .* 1e6, # Scale gradient to show dσ_z[mm]/dV[MV]
    errors_length .* 1e3 .* 1e6,
    param_name="RF Voltage [MV]",
    fom_name="Bunch Length [mm]"
)
savefig(p2, "rf_voltage_bunch_length.png")

println("Analysis complete. Results saved to rf_voltage_energy_spread.png and rf_voltage_bunch_length.png")

