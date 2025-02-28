"""
sensitivity.jl - Sensitivity calculation functions

This file implements the core sensitivity analysis algorithms that compute
derivatives of beam properties with respect to machine parameters.
"""

using Statistics
using StochasticAD

"""
    compute_sensitivity(
        transform::ParameterTransformation, 
        fom::FigureOfMerit, 
        param_value::Float64,
        initial_particles::StructArray{Particle{T}},
        sim_params::SimulationParameters;
        n_samples::Int=100
    ) where T<:Float64 -> ParameterSensitivity

Compute sensitivity using StochasticAD.
"""
function compute_sensitivity(
    transform::ParameterTransformation, 
    fom::FigureOfMerit, 
    param_value::Float64,
    initial_particles::StructArray{Particle{T}},
    sim_params::SimulationParameters;
    n_samples::Int=100
) where T<:Float64
    
    # Pre-allocate buffers
    n_particles = length(initial_particles)
    nbins = Int(n_particles/10)
    buffers = create_simulation_buffers(n_particles, nbins, T)
    
    # Define function for derivative estimation
    function sensitivity_fn(p)
        # Apply parameter transformation
        params_copy = apply_transform(transform, p, sim_params)
        
        # Use a fresh copy of particles each time to avoid state issues
        particles_copy = deepcopy(initial_particles)
        
        # Use a try-catch to diagnose any issues
        try
            # Run simulation
            results = longitudinal_evolve!(particles_copy, params_copy, buffers)
            
            # Compute figure of merit
            return compute_fom(fom, particles_copy, results)
        catch e
            # Print diagnostic information
            println("Error in sensitivity_fn with parameter value: ", p)
            println("Parameter type: ", typeof(p))
            println("Error: ", e)
            rethrow(e)
        end
    end
    
    # Collect samples with explicit diagnostics
    samples = Float64[]
    sizehint!(samples, n_samples)
    
    for i in 1:n_samples
        try
            sample = derivative_estimate(sensitivity_fn, param_value)
            push!(samples, sample)
            # Print the first few samples to verify we're getting non-zero values
            if i <= 3
                println("Sample $i gradient: $sample")
            end
        catch e
            println("Error in sample $i: ", e)
            # Continue with other samples
        end
    end
    
    if isempty(samples)
        error("All sensitivity samples failed")
    end
    
    # Compute statistics
    mean_derivative = mean(samples)
    uncertainty = std(samples) / sqrt(length(samples))
    
    println("Mean gradient: $mean_derivative ± $uncertainty")
    
    # Create and return sensitivity object
    return ParameterSensitivity(
        transform,
        fom,
        param_value,
        mean_derivative,
        uncertainty,
        samples
    )
end

"""
    autograd_statistics(func, param_value; n_samples=100)

Compute statistics for derivatives of a function.
"""
function autograd_statistics(func, param_value; n_samples=100)
    # Collect derivative samples
    samples = [derivative_estimate(func, param_value) for _ in 1:n_samples]
    
    # Compute statistics
    mean_derivative = mean(samples)
    uncertainty = std(samples) / sqrt(n_samples)
    
    return mean_derivative, uncertainty, samples
end