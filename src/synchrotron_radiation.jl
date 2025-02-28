"""
    synchrotron_radiation!(
        E0::T, 
        radius::T, 
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply synchrotron radiation damping to all particles.
"""
function synchrotron_radiation!(
    E0::T, 
    radius::T, 
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    # Calculate damping coefficient
    ∂U_∂E::T = 4 * 8.85e-5 * (E0/1e9)^3 / radius
    damping_factor::T = 1 - ∂U_∂E
    
    # Apply damping to each particle
    n_particles = length(particles)
    @inbounds for i in 1:n_particles
        particles.coordinates.ΔE[i] *= damping_factor
    end
    
    return nothing
end