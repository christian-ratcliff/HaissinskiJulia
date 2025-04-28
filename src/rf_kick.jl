using StochasticAD
using ThreadsX
using FLoops
# using StochasticAD

"""
    rf_kick!(
        voltage,
        sin_ϕs,
        rf_factor,
        ϕs,
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply RF cavity voltage to all particles.
Enhanced implementation for proper StochasticTriple propagation.
"""
function rf_kick!(
    voltage,
    sin_ϕs,
    rf_factor,
    ϕs,
    particles::StructArray{Particle{T}},
    buffers::SimulationBuffers{T}
) where T<:Float64
    
    # # Check if voltage is a StochasticTriple
    # is_stochastic = typeof(voltage) <: StochasticTriple
    
    # # Create a single wrapper function that will properly propagate the StochasticTriple
    # if is_stochastic
    #     # Define the core kick calculation function
    #     function apply_kick(v)
    #         for i in 1:length(particles)
    #             ϕ_val = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
    #             sin_term = sin(ϕ_val) - sin_ϕs
    #             # Apply kick
    #             particles.coordinates.ΔE[i] += v * sin_term
    #         end
    #         # Return a scalar result to ensure gradient propagation
    #         return sum(particles.coordinates.ΔE) / length(particles)
    #     end
        
    #     # Use propagate with the entire function to maintain StochasticTriple lineage
    #     # This scalar result isn't used, but ensures gradient propagation
    #     _ = StochasticAD.propagate(apply_kick, voltage)
    # else
    #     # Standard implementation for non-StochasticTriple case
    #     # for i in 1:length(particles)
    #     #     ϕ_val = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
    #     #     particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    #     # # end
    #     # z_vals = copy(particles.coordinates.z)  # Cache the value
    #     # ΔE_vals = copy(particles.coordinates.ΔE)  # Cache the value
    #     # sinϕ = sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs
    #     # particles.coordinates.ΔE .= particles.coordinates.ΔE .+ voltage .* (sin.(-particles.coordinates.z .* rf_factor .+ ϕs) .- sin_ϕs)
    #     # particles.coordinates.ΔE .= ΔE_vals  # Store the updated value

    #     for i in 1:length(particles)
    #         ϕ_val = -particles.coordinates.z[i] * rf_factor + ϕs
    #         particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    #     end
    # end

    # chunk_size = max(1, length(particles) ÷ Threads.nthreads())
        
    # Threads.@threads for chunk_start in 1:chunk_size:length(particles)
    #     chunk_end = min(chunk_start + chunk_size - 1, length(particles))
        
    #     @turbo for i in chunk_start:chunk_end
    #         ϕ_val = -particles.coordinates.z[i] * rf_factor + ϕs
    #         particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    #     end
    # end

    # Threads.@threads for tid in 1:Threads.nthreads()
    #     chunk_range = buffers.thread_chunks[tid]
    #     @turbo for i in chunk_range
    #         ϕ_val = -particles.coordinates.z[i] * rf_factor + ϕs
    #         particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    #     end
    # end

    @turbo for i in 1:length(particles.coordinates.z)
        ϕ_val = -particles.coordinates.z[i] * rf_factor + ϕs
        particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    end

    # ThreadsX.foreach(1:length(particles)) do i
    #     @inbounds begin
    #         z_i = particles.coordinates.z[i]
    #         ΔE_i = particles.coordinates.ΔE[i]
    
    #         ϕ_val = -z_i * rf_factor + ϕs
    #         ΔE_i += voltage * (sin(ϕ_val) - sin_ϕs)
    
    #         particles.coordinates.ΔE[i] = ΔE_i
    #     end
    # end

    # @floop ThreadedEx() for i in 1:length(particles)
    #     @inbounds begin # Use @inbounds for direct array access
    #         # Read values
    #         z_i = particles.coordinates.z[i]
    #         ΔE_i = particles.coordinates.ΔE[i] # Read initial ΔE

    #         # Calculate new value
    #         ϕ_val = -z_i * rf_factor + ϕs
    #         ΔE_new = ΔE_i + voltage * (sin(ϕ_val) - sin_ϕs)

    #         # Write back
    #         particles.coordinates.ΔE[i] = ΔE_new
    #     end
    # end


    
    return nothing
end