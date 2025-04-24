"""
wakefield.jl - Wakefield and collective effects

This file implements the wakefield calculations and collective effects
that couple particles within the beam, including:
- Wake function calculation
- Convolution with charge distribution
- Application of wakefield forces to particles
"""

using LoopVectorization
using FFTW
using Interpolations
using FHist

"""
    apply_wakefield_inplace!(
        particles::StructArray{Particle{T}},
        buffers::SimulationBuffers{T},
        wake_factor::T,
        wake_sqrt::T,
        cτ::T,
        current::T,
        σ_z::T,
        bin_edges
    ) where T<:Float64 -> Nothing

Apply wakefield effects to all particles.
"""
function apply_wakefield_inplace!(
    particles::StructArray{Particle{T}},
    buffers::SimulationBuffers{T},
    wake_factor::T,
    wake_sqrt::T,
    cτ::T,
    current::T,
    σ_z::T,
    bin_edges
    ) where T<:Float64
    
    # Clear buffers
    fill!(buffers.λ, zero(T))
    fill!(buffers.WF_temp, zero(T))
    fill!(buffers.convol, zero(Complex{T}))
    
    # Get z positions
    n_particles = length(particles)
    inv_cτ::T = 1 / cτ
    
    # Calculate histogram
    bin_centers::Vector{T}, bin_amounts::Vector{T} = calculate_histogram(particles.coordinates.z, bin_edges)
    nbins::Int = length(bin_centers)
    power_2_length::Int = nbins * 2
    
    
    # Calculate line charge density using Gaussian smoothing
    delta_std::T = (15 * σ_z) / σ_z / 100
    @inbounds for i in eachindex(bin_centers)
        buffers.λ[i] = delta(bin_centers[i], delta_std)
    end
    
    # Calculate wake function for each bin
    @inbounds for i in eachindex(bin_centers)
        z = bin_centers[i]
        buffers.WF_temp[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
    end

    # Prepare arrays for convolution
    normalized_amounts = bin_amounts .* (1/n_particles)
    λ = buffers.λ[1:nbins]
    WF_temp = buffers.WF_temp[1:nbins]
    convol = buffers.convol[1:power_2_length]
    
    # Perform convolution and scale by current
    convol .= FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length) .* current

    
    
    # Interpolate results back to particle positions
    temp_z = range(minimum(particles.coordinates.z), maximum(particles.coordinates.z), length=length(convol))
    resize!(buffers.potential, length(particles.coordinates.z))
    
    # Create an interpolation function
    # itp = LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line())
    copyto!(buffers.temp_ΔE[1:length(convol)], real.(convol))
    itp = LinearInterpolation(temp_z, buffers.temp_ΔE[1:length(convol)], extrapolation_bc=Line())
    
    # Apply the interpolated potential to particles
    @inbounds for i in eachindex(particles.coordinates.z)
        z = particles.coordinates.z[i]
        potential_value = itp(z)
        particles.coordinates.ΔE[i] -= potential_value
        # buffers.WF[i] = calculate_wake_function(z, wake_factor, wake_sqrt, inv_cτ)
    end
    
    return nothing
end

"""
    calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, cτ::T) where T<:Float64 -> T

Calculate the wake function for a given longitudinal position.
"""
function calculate_wake_function(z::T, wake_factor::T, wake_sqrt::T, inv_cτ::T) where T<:Float64
    return z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
end

# """
#     compute_line_charge_density(particles.coordinates.z::Vector{T}, σ_z::T, bin_edges) where T<:Float64 
#                                -> Tuple{Vector{T}, Vector{T}}

# Compute the line charge density distribution for a particle distribution.
# """
# function compute_line_charge_density(particles::StructArray{Particle{T}}, σ_z::T, bin_edges) where T<:Float64
#     # Calculate histogram
#     bin_centers, bin_amounts = calculate_histogram(particles.coordinates.z, bin_edges)
    
#     # Normalize
#     n_particles = length(particles.coordinates.z)
#     normalized_amounts = bin_amounts .* (1/n_particles)
    
#     # Apply Gaussian smoothing
#     delta_std = (15 * σ_z) / σ_z / 100
#     smoothed_density = Vector{T}(undef, length(bin_centers))
    
#     for i in eachindex(bin_centers)
#         smoothed_density[i] = delta(bin_centers[i], delta_std) * normalized_amounts[i]
#     end
    
#     return bin_centers, smoothed_density
# end