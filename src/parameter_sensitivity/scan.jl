using Base.Threads

"""
    scan_parameter(
        transform::ParameterTransformation, 
        fom::FigureOfMerit, 
        param_range::AbstractVector{Float64},
        initial_particles::StructArray{Particle{T}},
        sim_params::SimulationParameters;
        n_samples::Int=20
    ) where T<:Float64 -> Tuple{AbstractVector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Scan a parameter across a range of values using StochasticAD exclusively.
"""
function scan_parameter(
    transform::ParameterTransformation, 
    fom::FigureOfMerit, 
    param_range::AbstractVector{Float64},
    initial_particles::StructArray{Particle{T}},
    sim_params::SimulationParameters;
    n_samples::Int=20
) where T<:Float64
    
    # Pre-allocate result arrays
    n_points = length(param_range)
    fom_values = Vector{Float64}(undef, n_points)
    gradient_values = Vector{Float64}(undef, n_points)
    gradient_errors = Vector{Float64}(undef, n_points)
    
    # Process parameter points sequentially for better debugging
    println("Processing $n_points parameter points, using StochasticAD exclusively")
    
    # Define a function to process a single parameter value
    function process_parameter(i, param_value)
        try
            println("\n--- Starting parameter point $i/$n_points: value = $param_value ---")
            
            # Calculate actual FoM value
            n_particles = length(initial_particles)
            nbins = Int(n_particles/10)
            buffers = create_simulation_buffers(n_particles, nbins, T)
            
            # Create parameter copy for baseline evaluation
            params_copy = apply_transform(transform, param_value, sim_params)
            
            # Ensure reasonable n_turns value for sensitivity analysis
            original_n_turns = params_copy.n_turns
            if original_n_turns < 100
                println("Increasing n_turns to improve sensitivity detection.")
                # Create modified parameters with more turns
                params_copy = SimulationParameters(
                    params_copy.E0,
                    params_copy.mass,
                    params_copy.voltage,
                    params_copy.harmonic,
                    params_copy.radius,
                    params_copy.pipe_radius,
                    params_copy.α_c,
                    params_copy.ϕs,
                    params_copy.freq_rf,
                    max(100, original_n_turns),  # At least 100 turns
                    params_copy.use_wakefield,
                    params_copy.update_η,
                    params_copy.update_E0,
                    params_copy.SR_damping,
                    params_copy.use_excitation
                )
                println("Increased n_turns to $(params_copy.n_turns)")
            end
            
            # Run base simulation
            particles_copy = deepcopy(initial_particles)
            results = longitudinal_evolve!(particles_copy, params_copy, buffers)
            fom_value = compute_fom(fom, particles_copy, results)
            
            # Calculate sensitivity with StochasticAD
            sensitivity = compute_sensitivity(
                transform,
                fom,
                param_value,
                initial_particles,
                params_copy;
                n_samples=n_samples
            )
            
            # Store results
            fom_values[i] = fom_value
            gradient_values[i] = sensitivity.mean_derivative
            gradient_errors[i] = sensitivity.uncertainty
            
            # Update progress
            println("\n=== Completed parameter point $i/$n_points ===")
            println("Parameter value: $param_value")
            println("FoM value: $fom_value")
            println("Gradient: $(sensitivity.mean_derivative) ± $(sensitivity.uncertainty)")
            
            return true
        catch e
            println("\n!!! Error processing parameter point $i/$n_points: value = $param_value !!!")
            println("Error: $e")
            
            # Attempt with modified parameters 
            try
                println("Attempting with reduced complexity...")
                
                # Create simpler parameters (no wakefield, fewer turns)
                simpler_params = SimulationParameters(
                    sim_params.E0,
                    sim_params.mass,
                    param_value,  # Direct parameter value
                    sim_params.harmonic,
                    sim_params.radius,
                    sim_params.pipe_radius,
                    sim_params.α_c,
                    sim_params.ϕs,
                    sim_params.freq_rf,
                    50,  # Much fewer turns
                    false,  # No wakefield
                    true,   # Keep other settings
                    false,  # No E0 updates
                    true,   
                    false   # No excitation
                )
                
                # Define a simpler sensitivity function
                function simple_sensitivity_fn(p)
                    # Apply parameter transformation
                    params = apply_transform(transform, p, simpler_params)
                    
                    # Fresh particles
                    particles_simple = deepcopy(initial_particles)
                    buffers_simple = create_simulation_buffers(n_particles, nbins, T)
                    
                    # Run simulation
                    results = longitudinal_evolve!(particles_simple, params, buffers_simple)
                    
                    # Return FoM
                    return compute_fom(fom, particles_simple, results)
                end
                
                # Use StochasticAD with explicit algorithm
                simplified_samples = [
                    derivative_estimate(
                        simple_sensitivity_fn, 
                        param_value,
                        StochasticAD.ForwardAlgorithm(PrunedFIsBackend())
                    ) 
                    for _ in 1:5  # Reduced sample count
                ]
                
                # Calculate FoM value
                particles_simple = deepcopy(initial_particles)
                buffers_simple = create_simulation_buffers(n_particles, nbins, T)
                simple_params = apply_transform(transform, param_value, simpler_params)
                simple_results = longitudinal_evolve!(particles_simple, simple_params, buffers_simple)
                simple_fom = compute_fom(fom, particles_simple, simple_results)
                
                # Get stats
                simplified_sensitivity = mean(simplified_samples)
                simplified_uncertainty = std(simplified_samples) / sqrt(length(simplified_samples))
                
                println("Simplified StochasticAD: $simplified_sensitivity ± $simplified_uncertainty")
                
                fom_values[i] = simple_fom
                gradient_values[i] = simplified_sensitivity
                gradient_errors[i] = simplified_uncertainty
                
                return true
            catch e_fallback
                println("Fallback StochasticAD calculation failed: ", e_fallback)
                
                # If all StochasticAD approaches fail, use NaN to indicate failure
                fom_values[i] = NaN
                gradient_values[i] = NaN
                gradient_errors[i] = NaN
                return false
            end
        end
    end
    
    # Process all parameter points
    for i in 1:n_points
        process_parameter(i, param_range[i])
    end
    
    # Print summary
    println("\n=== Parameter scan complete ===")
    for i in 1:n_points
        println("Parameter $i: value = $(param_range[i]), FoM = $(fom_values[i]), Gradient = $(gradient_values[i]) ± $(gradient_errors[i])")
    end
    
    return param_range, fom_values, gradient_values, gradient_errors
end