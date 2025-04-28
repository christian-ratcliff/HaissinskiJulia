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
    particles::StructArray{Particle{T}},
    buffers::SimulationBuffers{T}
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
    
    # Threads.@threads for tid in 1:Threads.nthreads()
    #     chunk_range = buffers.thread_chunks[tid]
    #     @turbo for i in chunk_range
    #         particles.coordinates.ΔE[i] *= damping_factor
    #     end
    # end

    @turbo for i in 1:length(particles.coordinates.z)
        particles.coordinates.ΔE[i] *= damping_factor
    end
    
    return nothing
end