using StochasticAD
"""
    synchrotron_radiation!(
        E0, 
        radius, 
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply synchrotron radiation damping to all particles.
Memory-efficient implementation for StochasticTriple.
"""
function synchrotron_radiation!(
    E0, 
    radius, 
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Check type once, outside the loop
    # is_stochastic = (typeof(E0) <: StochasticTriple) || (typeof(radius) <: StochasticTriple)
    
    # if is_stochastic
    #     # Calculate damping and apply to all particles
    #     damping_fn = (e0, r) -> begin
    #         ∂U_∂E = 4 * 8.85e-5 * (e0/1e9)^3 / r
    #         return 1 - ∂U_∂E
    #     end
        
    #     # Get the damping factor - single propagate call
    #     damping_factor = StochasticAD.propagate(damping_fn, E0, radius)
        
    #     # Apply to all particles (vectorized)
    #     particles.coordinates.ΔE .*= damping_factor
    # else
    #     # Standard implementation 
    #     ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
    #     damping_factor = 1 - ∂U_∂E
        
    #     # Apply damping
    #     particles.coordinates.ΔE .*= damping_factor
    # end

    # Standard implementation 
    ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
    damping_factor = 1 - ∂U_∂E
    
    # # Apply damping in parallel chunks
    # chunk_size = max(1, length(particles) ÷ Threads.nthreads() ÷ 4)
    
    # Threads.@threads for chunk_start in 1:chunk_size:length(particles)
    #     chunk_end = min(chunk_start + chunk_size - 1, length(particles))
        
    #     @turbo for i in chunk_start:chunk_end
    #         particles.coordinates.ΔE[i] *= damping_factor
    #     end
    # end

    particles.coordinates.ΔE .*= damping_factor
    
    return nothing
end