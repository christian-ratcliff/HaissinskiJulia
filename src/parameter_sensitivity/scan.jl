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

Scan a parameter across a range of values using parallel computation.
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
    
    # Process parameter points in parallel
    println("Processing $n_points parameter points with $(Threads.nthreads()) threads")
    progress_lock = ReentrantLock()
    completed = Threads.Atomic{Int}(0)
    
    # Define a function to process a single parameter value
    function process_parameter(i, param_value)
        try
            println("\n--- Starting parameter point $i/$n_points: value = $param_value ---")
            
            # Calculate actual FoM value
            n_particles = length(initial_particles)
            nbins = Int(n_particles/10)
            buffers = create_simulation_buffers(n_particles, nbins, T)
            
            params_copy = apply_transform(transform, param_value, sim_params)
            particles_copy = deepcopy(initial_particles)
            results = longitudinal_evolve!(particles_copy, params_copy, buffers)
            fom_value = compute_fom(fom, particles_copy, results)
            
            # Calculate sensitivity
            sensitivity = compute_sensitivity(
                transform,
                fom,
                param_value,
                initial_particles,
                sim_params;
                n_samples=n_samples
            )
            
            # Store results
            fom_values[i] = fom_value
            gradient_values[i] = sensitivity.mean_derivative
            gradient_errors[i] = sensitivity.uncertainty
            
            # Update progress
            new_completed = Threads.atomic_add!(completed, 1)
            lock(progress_lock) do
                println("\n=== Completed parameter point $i/$n_points ($(new_completed+1)/$n_points total) ===")
                println("Parameter value: $param_value")
                println("FoM value: $fom_value")
                println("Gradient: $(sensitivity.mean_derivative) ± $(sensitivity.uncertainty)")
            end
            
            return true
        catch e
            lock(progress_lock) do
                println("\n!!! Error processing parameter point $i/$n_points: value = $param_value !!!")
                println("Error: $e")
            end
            return false
        end
    end
    
    # Process all parameter points in parallel
    Threads.@threads for i in 1:n_points
        process_parameter(i, param_range[i])
    end
    
    # Print summary
    println("\n=== Parameter scan complete ===")
    for i in 1:n_points
        println("Parameter $i: value = $(param_range[i]), FoM = $(fom_values[i]), Gradient = $(gradient_values[i]) ± $(gradient_errors[i])")
    end
    
    return param_range, fom_values, gradient_values, gradient_errors
end