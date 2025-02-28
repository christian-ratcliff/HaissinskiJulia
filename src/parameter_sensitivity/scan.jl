"""
scan.jl - Parameter scanning functionality

This file implements parameter scanning functionality to explore
how beam properties vary across a range of parameter values.
"""


"""
    scan_parameter(
        transform::ParameterTransformation, 
        fom::FigureOfMerit, 
        param_range::AbstractVector{Float64},
        initial_particles::StructArray{Particle{T}},
        sim_params::SimulationParameters{T};
        n_samples::Int=20
    ) where T<:Float64 -> Tuple{AbstractVector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Scan a parameter across a range of values, computing the FoM and its gradient at each point.
"""
function scan_parameter(
    transform::ParameterTransformation, 
    fom::FigureOfMerit, 
    param_range::AbstractVector{Float64},
    initial_particles::StructArray{Particle{T}},
    sim_params::SimulationParameters{T};
    n_samples::Int=20
) where T<:Float64
    
    # Pre-allocate result arrays
    n_points = length(param_range)
    fom_values = Vector{Float64}(undef, n_points)
    gradient_values = Vector{Float64}(undef, n_points)
    gradient_errors = Vector{Float64}(undef, n_points)
    
    # Pre-allocate buffers (reused for efficiency)
    n_particles = length(initial_particles)
    nbins = Int(n_particles/10)
    buffers = create_simulation_buffers(n_particles, nbins, T)
    
    # For each parameter value
    for (i, param_value) in enumerate(param_range)
        # Calculate actual FoM value
        params_copy = apply_transform(transform, param_value, sim_params)
        particles_copy = deepcopy(initial_particles)
        results = longitudinal_evolve!(particles_copy, params_copy, buffers)
        fom_values[i] = compute_fom(fom, particles_copy, results)
        
        # Calculate sensitivity
        sensitivity = compute_sensitivity(
            transform,
            fom,
            param_value,
            initial_particles,
            sim_params,
            n_samples=n_samples
        )
        
        gradient_values[i] = sensitivity.mean_derivative
        gradient_errors[i] = sensitivity.uncertainty
    end
    
    return param_range, fom_values, gradient_values, gradient_errors
end