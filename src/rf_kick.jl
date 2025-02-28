using StochasticAD
"""
    rf_kick!(
        voltage,
        sin_ϕs,
        rf_factor,
        ϕs,
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply RF cavity voltage to all particles.
Memory-efficient implementation for StochasticTriple.
"""
function rf_kick!(
    voltage,
    sin_ϕs,
    rf_factor,
    ϕs,
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Check type once, outside the loop
    is_stochastic = typeof(voltage) <: StochasticTriple
    
    # Pre-compute the constant values
    rf_factor_val = convert(T, rf_factor)
    ϕs_val = convert(T, ϕs)
    sin_ϕs_val = convert(T, sin_ϕs)
    
    # Process particles based on type
    if is_stochastic
        # For StochasticTriple, we need special handling
        # But minimize allocations inside the loop
        
        # Extract the raw values for faster computation
        # z_values = particles.coordinates.z
        # ΔE_values = particles.coordinates.ΔE
        
        # Process particles
        for i in 1:length(particles)
            # Calculate phase and sine term
            ϕ_val = z_to_ϕ(particles.coordinates.z[i], rf_factor_val, ϕs_val)
            sin_term = sin(ϕ_val) - sin_ϕs_val
            
            # Propagate through the operation with minimal intermediate variables
            # This is a single StochasticTriple operation for the whole batch
            ΔE_values[i] = StochasticAD.propagate(
                v -> particles.coordinates.ΔE[i] + v * sin_term, 
                voltage
            )
        end
    else
        # Standard Float64 implementation - vectorized when possible
        for i in 1:length(particles)
            ϕ_val = z_to_ϕ(particles.coordinates.z[i], rf_factor_val, ϕs_val)
            particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs_val)
        end
    end
end
