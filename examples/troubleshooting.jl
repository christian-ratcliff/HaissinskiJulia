# """
# ultra_simple_test.jl - Extremely simplified test for StochasticAD integration

# This script creates the most basic test possible to verify StochasticAD functionality
# with the beam dynamics simulation, while avoiding all possible complexities.
# """

# include("../src/StochasticHaissinski.jl")

# begin
#     using .StochasticHaissinski
#     using Plots
#     using Statistics
#     using StochasticAD
#     using Random
# end;

# # Set seed for reproducibility
# Random.seed!(123456);

# # Verify basic StochasticAD functionality
# function test_basic_stochastic_ad()
#     println("Testing basic StochasticAD functionality...")
#     f(x) = x^2 
#     x_val = 3.0
    
#     # Known derivative
#     true_deriv = 2 * x_val
    
#     # Test with StochasticAD
#     samples = [derivative_estimate(f, x_val) for _ in 1:10]
#     est_deriv = mean(samples)
    
#     println("True derivative: $true_deriv")
#     println("StochasticAD estimate: $est_deriv")
    
#     return abs(est_deriv - true_deriv) < 0.1
# end

# # Test the most minimal possible function involving RF voltage
# function create_minimal_test()
#     # Create the most basic test case possible
#     println("\nCreating minimal RF voltage sensitivity test...")
    
#     # Function that mimics the effect of RF voltage on energy spread
#     # This is a greatly simplified model of what happens in the real simulation
#     function minimal_voltage_effect(voltage)
#         # Very basic model: Energy spread ~ sqrt(voltage)
#         energy_spread = sqrt(voltage * 1e-6) * 1e6
#         return energy_spread
#     end
    
#     # Known analytical derivative: d/dV (sqrt(V*1e-6)*1e6) = 0.5 * 1e6/sqrt(V*1e-6)
#     voltage_test = 5.0e6  # 5 MV
#     analytical_derivative = 0.5 * 1e6 / sqrt(voltage_test * 1e-6)
    
#     println("Analytical derivative at V=$voltage_test: $analytical_derivative")
    
#     # Test with StochasticAD
#     samples = [derivative_estimate(minimal_voltage_effect, voltage_test) for _ in 1:20]
#     est_deriv = mean(samples)
#     est_error = std(samples) / sqrt(20)
    
#     println("StochasticAD estimate: $est_deriv ± $est_error")
#     println("Relative error: $(abs(est_deriv - analytical_derivative)/analytical_derivative * 100)%")
    
#     # Now create a slightly more complex function that's closer to the real simulation
#     function slightly_more_complex(voltage)
#         # Initial energy
#         E0 = 4e9  # 4 GeV
        
#         # Energy gain per turn
#         # In the real simulation, this happens in multiple small steps
#         sin_phase = 0.5  # Approximation of sin(sync_phase)
#         energy_gain = voltage * sin_phase
        
#         # Final energy after 10 turns
#         E_final = E0
#         for _ in 1:10
#             E_final += energy_gain
#         end
        
#         # Energy spread is approximately proportional to sqrt(E_final)
#         # This is a very rough approximation of the physical model
#         energy_spread = sqrt(E_final) * 1e-3
        
#         return energy_spread
#     end
    
#     println("\nTesting slightly more complex model...")
    
#     # Test with StochasticAD
#     complex_samples = [derivative_estimate(slightly_more_complex, voltage_test) for _ in 1:20]
#     complex_deriv = mean(complex_samples)
#     complex_error = std(complex_samples) / sqrt(20)
    
#     println("StochasticAD complex model estimate: $complex_deriv ± $complex_error")
    
#     return true
# end

# # Set up extremely simple simulation parameters
# function run_simplified_simulation()
#     println("\nRunning extremely simplified simulation...")
    
#     # Basic physical parameters
#     E0_ini = 4e9  # 4 GeV
#     mass = MASS_ELECTRON
#     harmonic = 100
#     radius = 100.0
#     volt_base = 5e6  # 5 MV
#     α_c = 1e-4
#     ϕs = π/2  # 90 degrees - simplifies sin(ϕs) = 1
    
#     # Derive other parameters
#     γ = E0_ini/mass
#     β = sqrt(1 - 1/γ^2)
#     freq_rf = β * SPEED_LIGHT / (2π * radius / harmonic)
    
#     # Distribution parameters
#     μ_z = 0.0
#     μ_E = 0.0
#     σ_E0 = 1e6  # 1 MeV energy spread
#     σ_z0 = 0.003  # 3 mm bunch length

#     # Create ULTRA simplified parameters - bare minimum for test
#     base_params = SimulationParameters(
#         E0_ini,      # E0
#         mass,        # mass
#         volt_base,   # voltage
#         harmonic,    # harmonic
#         radius,      # radius
#         0.001,       # pipe_radius
#         α_c,         # α_c
#         ϕs,          # ϕs
#         freq_rf,     # freq_rf
#         5,           # n_turns (absolutely minimal)
#         false,       # use_wakefield
#         false,       # update_η
#         false,       # update_E0
#         true,       # SR_damping
#         true        # use_excitation
#     )
    
#     # Create transformation for voltage
#     voltage_transform = VoltageTransform()
    
#     # Generate a very small number of test particles
#     n_particles = Int64(1e2)  # Just 100 particles
#     particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
#     println("Initial beam parameters: σ_E = $σ_E eV, σ_z = $σ_z m")
    
#     # Create a specialized sensitivity function just for testing
#     function simple_voltage_sensitivity(voltage)
#         # Create parameters with the specified voltage
#         simple_params = SimulationParameters(
#             base_params.E0,
#             base_params.mass,
#             voltage,  # Use the voltage parameter
#             base_params.harmonic,
#             base_params.radius,
#             base_params.pipe_radius,
#             base_params.α_c,
#             base_params.ϕs,
#             base_params.freq_rf,
#             base_params.n_turns,
#             base_params.use_wakefield,
#             base_params.update_η,
#             base_params.update_E0,
#             base_params.SR_damping,
#             base_params.use_excitation
#         )
        
#         # Create fresh particles for each run
#         test_particles = deepcopy(particles)
        
#         # Create buffers
#         nbins = Int(n_particles/10)
#         test_buffers = create_simulation_buffers(n_particles, nbins, Float64)
        
#         # Run simulation
#         σ_E, σ_z, E0_final = longitudinal_evolve!(test_particles, simple_params, test_buffers)
        
#         # Return energy spread as our figure of merit
#         println("Voltage: $voltage, σ_E: $σ_E")
#         return σ_E
#     end
    
#     # Test the function directly
#     baseline = simple_voltage_sensitivity(volt_base)
    
#     # Test with a perturbation for numerical derivative
#     delta = volt_base * 0.01
#     perturbed = simple_voltage_sensitivity(volt_base + delta)
#     numerical_deriv = (perturbed - baseline) / delta
    
#     println("Numerical derivative: $numerical_deriv")
    
#     # Now try with StochasticAD
#     println("\nTesting with StochasticAD...")
    
#     try
#         # Just a single test to keep it simple
#         stoch_deriv = derivative_estimate(simple_voltage_sensitivity, volt_base)
#         println("StochasticAD derivative: $stoch_deriv")
#         return true
#     catch e
#         println("StochasticAD test failed: $e")
#         return false
#     end
# end

# # Run the tests in sequence
# println("=== Basic StochasticAD test ===")
# if test_basic_stochastic_ad()
#     println("Basic StochasticAD test PASSED")
    
#     println("\n=== Minimal voltage effect test ===")
#     if create_minimal_test()
#         println("Minimal voltage effect test PASSED")
        
#         println("\n=== Simplified simulation test ===")
#         if run_simplified_simulation()
#             println("Simplified simulation test PASSED")
#             println("\nAll tests PASSED! StochasticAD is working correctly with your simulation.")
#         else
#             println("Simplified simulation test FAILED")
#         end
#     else
#         println("Minimal voltage effect test FAILED")
#     end
# else
#     println("Basic StochasticAD test FAILED")
# end

# """
# stochastic_test.jl - Test with proper randomness for StochasticAD

# This script enables the quantum excitation and other stochastic processes
# required for StochasticAD to properly compute gradients.
# """

# include("../src/StochasticHaissinski.jl")

# begin
#     using .StochasticHaissinski
#     using Plots
#     using Statistics
#     using StochasticAD
#     using Random
# end;

# # Set seed for reproducibility 
# Random.seed!(1234);

# # Test StochasticAD on a simple random function
# function test_basic_stochastic_ad()
#     println("=== Testing StochasticAD with randomness ===")
    
#     # Simple function with randomness
#     function random_function(p)
#         # This is a simple function with randomness
#         # StochasticAD tracks how p affects the expected value
#         return p * rand() + p^2
#     end
    
#     p_test = 2.0
#     # True derivative of E[p * rand() + p^2] = E[rand()] + 2p = 0.5 + 2p = 4.5
    
#     # Test with StochasticAD
#     n_samples = 1000
#     samples = [derivative_estimate(random_function, p_test) for _ in 1:n_samples]
#     est_deriv = mean(samples)
#     est_error = std(samples) / sqrt(n_samples)
    
#     println("StochasticAD with randomness: $est_deriv ± $est_error")
#     println("Expected: approximately 4.5")
    
#     return true
# end

# # Run a simple simulation with quantum excitation enabled
# function run_stochastic_simulation()
#     println("\n=== Running simulation with quantum excitation ===")
    
#     # Physical parameters
#     E0_ini = 4e9
#     mass = MASS_ELECTRON
#     voltage = 5e6
#     harmonic = 360
#     radius = 250.0
#     pipe_radius = 0.00025
#     α_c = 3.68e-4
#     γ = E0_ini/mass
#     β = sqrt(1 - 1/γ^2)
#     ϕs = 5π/6
#     freq_rf = β * SPEED_LIGHT / (2π * radius / harmonic)
#     sin_ϕs = sin(ϕs)
    
#     # Distribution parameters
#     μ_z = 0.0
#     μ_E = 0.0
#     σ_E0 = 1e6
#     σ_z0 = 0.003
    
#     # Create simulation parameters WITH RANDOMNESS ENABLED
#     base_params = SimulationParameters(
#         E0_ini,      # E0
#         mass,        # mass
#         voltage,     # voltage
#         harmonic,    # harmonic
#         radius,      # radius
#         pipe_radius, # pipe_radius
#         α_c,         # α_c
#         ϕs,          # ϕs
#         freq_rf,     # freq_rf
#         20,          # n_turns
#         true,       # use_wakefield - not critical for this test
#         true,        # update_η 
#         false,        # update_E0
#         true,        # SR_damping
#         true         # use_excitation - CRITICAL FOR STOCHASTICAD!
#     )
    
#     # Generate particles
#     n_particles = Int64(1e3)
#     particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf);
#     println("Initial beam parameters: σ_E = $σ_E eV, σ_z = $σ_z m")
    
#     # Create RF voltage sensitivity test function
#     function voltage_sensitivity(v)
#         # Create parameters with the test voltage
#         test_params = SimulationParameters(
#             base_params.E0,
#             base_params.mass,
#             v,  # Test voltage
#             base_params.harmonic,
#             base_params.radius,
#             base_params.pipe_radius,
#             base_params.α_c,
#             base_params.ϕs,
#             base_params.freq_rf,
#             base_params.n_turns,
#             base_params.use_wakefield,
#             base_params.update_η,
#             base_params.update_E0,
#             base_params.SR_damping,
#             base_params.use_excitation  # KEEP RANDOMNESS ENABLED
#         )
        
#         # Fresh particles for each run
#         # test_particles = deepcopy(particles)
        
#         # Create buffers
#         nbins = Int(n_particles/10)
#         test_buffers = create_simulation_buffers(n_particles, nbins, Float64)
#         test_particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf)
#         # Run simulation
#         σ_E_final, σ_z_final, E0_final = longitudinal_evolve!(test_particles, test_params, test_buffers)
        
#         # Return energy spread as figure of merit
#         return σ_E_final
#     end
    
#     # Test with nominal value
#     v_test = voltage
#     nominal = voltage_sensitivity(v_test)
#     println("Nominal voltage ($v_test): Energy Spread = $nominal")
    
#     # Test with perturbed value for numerical derivative
#     delta = v_test * 0.01
#     perturbed = voltage_sensitivity(v_test + delta)
#     numerical_deriv = (perturbed - nominal) / delta
#     println("Perturbed voltage ($(v_test + delta)): Energy Spread = $perturbed")
#     println("Numerical derivative: $numerical_deriv")
    
#     # Now test with StochasticAD
#     println("\nTesting with StochasticAD (5 samples):")
#     for i in 1:5
#         # Use a new random seed for each test to see variation
#         Random.seed!(1234 + i)
#         deriv = derivative_estimate(voltage_sensitivity, v_test)
#         println("Sample $i: $deriv")
#     end
    
#     # Run more samples for statistical significance
#     println("\nRunning 20 samples for statistical significance...")
#     samples = []
#     for i in 1:20
#         Random.seed!(2000 + i)
#         deriv = derivative_estimate(voltage_sensitivity, v_test)
#         push!(samples, deriv)
#     end
    
#     mean_deriv = mean(samples)
#     std_deriv = std(samples)
#     println("Mean derivative: $mean_deriv ± $(std_deriv/sqrt(20))")
#     println("Numerical derivative: $numerical_deriv")
    
#     return true
# end

# # Run both tests
# if test_basic_stochastic_ad()
#     println("Basic StochasticAD test passed!")
    
#     # Run the simulation with randomness
#     if run_stochastic_simulation()
#         println("Stochastic simulation test complete!")
#     else
#         println("Stochastic simulation test failed!")
#     end
# else
#     println("Basic StochasticAD test failed!")
# end



include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using Plots
    using Statistics
    using StochasticAD
    using Random
end;

# Test StochasticAD with a simple random example
function verify_stochastic_ad()
    println("Verifying StochasticAD with randomness...")
    
    # Function with randomness
    function stochastic_example(p)
        # This returns p * U where U ~ Uniform(0,1)
        # The expected value is p * 0.5
        # The derivative of E[p*U] with respect to p is E[U] = 0.5
        abc = p^2
        return abc * rand()
    end
    
    stochastic_example(p::StochasticAD.StochasticTriple) = StochasticAD.propagate(stochastic_example, p)
    p_val = 5.0
    true_deriv = 0.5  # E[U] = 0.5 for U ~ Uniform(0,1)
    
    # Collect samples
    n_samples = 100
    samples = [derivative_estimate(stochastic_example, p_val) for _ in 1:n_samples]
    mean_deriv = mean(samples)
    std_deriv = std(samples) / sqrt(n_samples)
    
    println("Expected derivative: $true_deriv")
    println("StochasticAD: $mean_deriv ± $std_deriv")
    
    return abs(mean_deriv - true_deriv) < 3 * std_deriv
end

# Function to generate unique particle distribution for each run
function generate_unique_particles(params, μ_z, μ_E, σ_z0, σ_E0, n_particles)
    # Extract needed parameters
    E0 = params.E0
    mass = params.mass
    ϕs = params.ϕs
    freq_rf = params.freq_rf
    
    # Generate completely fresh particles
    particles, σ_E, σ_z, E0 = generate_particles(
        μ_z, μ_E, σ_z0, σ_E0, n_particles, E0, mass, ϕs, freq_rf
        )
    
    return particles, σ_E, σ_z
end

# Improved sensitivity calculation that properly captures stochasticity
function compute_true_stochastic_sensitivity(
    transform::VoltageTransform, 
    fom::EnergySpreadFoM, 
    param_value::Float64,
    base_params::SimulationParameters;
    n_samples::Int=20,
    n_particles::Int=1000,
    μ_z::Float64=0.0,
    μ_E::Float64=0.0,
    σ_z0::Float64=0.003,
    σ_E0::Float64=1.0e6
)
    
    # Define sensitivity function that generates new particles each time
    function stochastic_sensitivity_fn(p)
        # Apply parameter transformation
        params = apply_transform(transform, p, base_params)
        # println(params)
        # Generate completely new particles for this run
        particles, _, _ = generate_unique_particles(
            params, μ_z, μ_E, σ_z0, σ_E0, n_particles
            ) #This ordering is something that I might need to change for online tuning
        
        # Create buffers
        nbins = Int(n_particles/10)
        buffers = create_simulation_buffers(n_particles, nbins, Float64)
        
        # Run simulation
        results = longitudinal_evolve!(particles, params, buffers)
        
        # Compute figure of merit
        fom_value = compute_fom(fom, particles, results)
        # print(fom_value)
        return fom_value
    end
    
    # Generate baseline value
    Random.seed!(12345)
    baseline = stochastic_sensitivity_fn(param_value)
    println("Baseline FoM at $param_value: $baseline")
    
    # Collect StochasticAD samples
    println("Collecting $n_samples samples...")
    gradient_samples = Float64[]
    
    for i in 1:n_samples
        # Use different random seed for each sample
        Random.seed!(20000 + i)
        
        try
            # Compute gradient estimate
            # println(stochastic_sensitivity_fn(param_value))
            sample = derivative_estimate(stochastic_sensitivity_fn, param_value)
            push!(gradient_samples, sample)
            println("Sample $i: $sample")
        catch e
            println("Error in sample $i: $e")
        end
    end
    
    # Handle case where all samples failed
    if isempty(gradient_samples)
        error("All gradient samples failed")
    end
    
    # Compute statistics
    mean_gradient = mean(gradient_samples)
    uncertainty = std(gradient_samples) / sqrt(length(gradient_samples))
    
    println("Gradient: $mean_gradient ± $uncertainty")
    
    # Create and return sensitivity object
    return ParameterSensitivity(
        transform,
        fom,
        param_value,
        mean_gradient,
        uncertainty,
        gradient_samples
        )
end

# Improved scan function that regenerates particles for each parameter
function scan_stochastic_parameter(
    transform::VoltageTransform, 
    fom::EnergySpreadFoM, 
    param_range::AbstractVector{Float64},
    base_params::SimulationParameters;
    n_samples::Int=20,
    n_particles::Int=1000,
    μ_z::Float64=0.0,
    μ_E::Float64=0.0,
    σ_z0::Float64=0.005,
    σ_E0::Float64=1.0e6
    )
    # Pre-allocate result arrays
    n_points = length(param_range)
    fom_values = Vector{Float64}(undef, n_points)
    gradient_values = Vector{Float64}(undef, n_points)
    gradient_errors = Vector{Float64}(undef, n_points)
    
    # Process each parameter point
    for i in 1:n_points
        param_value = param_range[i]
        println("\n=== Parameter point $i/$n_points: $param_value ===")
        
        # Compute FoM at this parameter value
        params = apply_transform(transform, param_value, base_params)
        println("Parameters: ", params)
        # Generate particles for FoM calculation
        Random.seed!(50000 + i)
        particles, _, _ = generate_unique_particles(
            params, μ_z, μ_E, σ_z0, σ_E0, n_particles
        )
        
        # Create buffers and run simulation
        nbins = Int(n_particles/10)
        buffers = create_simulation_buffers(n_particles, nbins, Float64)
        results = longitudinal_evolve!(particles, params, buffers)
        
        # Compute and store FoM
        fom_value = compute_fom(fom, particles, results)
        # fom_value = results[1]
        fom_values[i] = fom_value
        
        # Compute sensitivity at this parameter value
        sensitivity = compute_true_stochastic_sensitivity(
            transform, fom, param_value, base_params;
            n_samples=n_samples,
            n_particles=n_particles,
            μ_z=μ_z, μ_E=μ_E, σ_z0=σ_z0, σ_E0=σ_E0
        )
        
        # Store gradient information
        gradient_values[i] = sensitivity.mean_derivative
        gradient_errors[i] = sensitivity.uncertainty
        
        println("Parameter $param_value: FoM = $fom_value, Gradient = $(sensitivity.mean_derivative) ± $(sensitivity.uncertainty)")
    end
    
    return param_range, fom_values, gradient_values, gradient_errors
end

# Set up physical parameters
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
    ϕs = 5π/6
    freq_rf = β * SPEED_LIGHT / (2π * radius / harmonic)
    
    # Distribution parameters
    μ_z = 0.0
    μ_E = 0.0
    σ_E0 = 1e6
    σ_z0 = 0.005
    
    # Create simulation parameters
    # Ensure all stochastic processes are enabled!
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
        1,          # n_turns - reduced for testing
        true,        # use_wakefield 
        true,        # update_η
        false,        # update_E0
        true,        # SR_damping
        true         # use_excitation
    )
end;

# Verify StochasticAD works properly
if !verify_stochastic_ad()
    error("StochasticAD verification failed")
else
    println("StochasticAD verification passed")
end

# Set up parameter transformations and figures of merit
voltage_transform = VoltageTransform()
energy_spread_fom = EnergySpreadFoM()
bunch_length_fom = BunchLengthFoM()

# Define voltage range for analysis
voltage_range = [4.0e6, 5.0e6, 6.0e6];  # Use few points for testing

# Test a single sensitivity calculation first
println("\n=== Testing single sensitivity calculation ===")
test_sensitivity = compute_true_stochastic_sensitivity(
    voltage_transform,
    energy_spread_fom,
    voltage,
    base_params,
    n_samples=5,  # Use few samples for testing
    n_particles=500  # Reduced for testing
)

# Run parameter scan if the test was successful
# if !isnan(test_sensitivity.mean_derivative)
#     # Scan voltage effect on energy spread
#     println("\n=== Scanning RF voltage effect on energy spread ===")
#     params_energy, foms_energy, grads_energy, errors_energy = scan_stochastic_parameter(
#         voltage_transform,
#         energy_spread_fom,
#         voltage_range,
#         base_params,
#         n_samples=10,
#         n_particles=500
#     )
    
#     # Scan voltage effect on bunch length
#     println("\n=== Scanning RF voltage effect on bunch length ===")
#     params_length, foms_length, grads_length, errors_length = scan_stochastic_parameter(
#         voltage_transform,
#         bunch_length_fom,
#         voltage_range,
#         base_params,
#         n_samples=10,
#         n_particles=500
#     )
    
#     # Plot results
#     println("\n=== Creating plots ===")
    
#     # Energy spread sensitivity plot
#     p1 = plot_sensitivity_scan(
#         params_energy ./ 1e6, 
#         foms_energy ./ 1e6, 
#         grads_energy ./ 1e6 .* 1e6, # Scale gradient to show dσ_E[MeV]/dV[MV]
#         errors_energy ./ 1e6 .* 1e6,
#         param_name="RF Voltage [MV]",
#         fom_name="Energy Spread [MeV]"
#     )
#     savefig(p1, "rf_voltage_energy_spread.png")
    
#     # Bunch length sensitivity plot
#     p2 = plot_sensitivity_scan(
#         params_length ./ 1e6, 
#         foms_length .* 1e3, # Convert to mm
#         grads_length .* 1e3 .* 1e6, # Scale gradient to show dσ_z[mm]/dV[MV]
#         errors_length .* 1e3 .* 1e6,
#         param_name="RF Voltage [MV]",
#         fom_name="Bunch Length [mm]"
#     )
#     savefig(p2, "rf_voltage_bunch_length.png")
    
#     println("Analysis complete. Results saved to rf_voltage_energy_spread.png and rf_voltage_bunch_length.png")
# else
#     println("Initial sensitivity test failed. Cannot proceed with parameter scanning.")
# end



function stochastic_sensitivity_fn(p, particles, base_params)
    # Apply parameter transformation
    transform = VoltageTransform()
    n_particles = 100
    params = apply_transform(transform, p, base_params)
    
    # println(params)
    # Generate completely new particles for this run
    # particles, _, _ = generate_unique_particles(
    #     params, μ_z, μ_E, σ_z0, σ_E0, n_particles
    #     ) #This ordering is something that I might need to change for online tuning
    # Random.seed!(12345)
    # Create buffers
    nbins = Int(n_particles/10)
    buffers = create_simulation_buffers(n_particles, nbins, Float64)
    
    # Run simulation
    results = longitudinal_evolve!(particles, params, buffers)
    
    # Compute figure of merit
    fom_value = compute_fom(EnergySpreadFoM(), particles, results)
    # print(fom_value)
    return fom_value
end

particles, _, _ = generate_unique_particles(
        base_params, μ_z, μ_E, σ_z0, σ_E0, 100
        );

derivative_estimate(p-> stochastic_sensitivity_fn(p, particles, base_params), 5.0e6)
stochastic_triple(p-> stochastic_sensitivity_fn(p, particles, base_params), 5.0e6)
StochasticAD.perturbations(stochastic_triple(stochastic_sensitivity_fn, 5.0e6))