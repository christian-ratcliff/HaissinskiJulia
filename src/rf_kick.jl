"""
    rf_kick!(
        voltage::T,
        sin_ϕs::T,
        rf_factor::T,
        ϕs::T,
        particles::StructArray{Particle{T}}
    ) where T<:Float64 -> Nothing

Apply RF cavity voltage to all particles.
"""
function rf_kick!(
    voltage::T,
    sin_ϕs::T,
    rf_factor::T,
    ϕs::T,
    particles::StructArray{Particle{T}}
) where T<:Float64
    
    @inbounds for i in 1:length(particles)
        ϕ_val = z_to_ϕ(particles.coordinates.z[i], rf_factor, ϕs)
        particles.coordinates.ΔE[i] += voltage * (sin(ϕ_val) - sin_ϕs)
    end
end