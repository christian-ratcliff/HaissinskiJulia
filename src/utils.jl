function threaded_fieldwise_copy!(destination, source)
    @assert length(destination.z) == length(source.z)
    @turbo for i in 1:length(source.z)
        destination.z[i] = source.z[i]
        destination.ΔE[i] = source.ΔE[i]
        destination.ϕ[i] = source.ϕ[i]
    end
end



function assign_to_turn!(particle_trajectory, particle_states, turn)
    threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
end

@inline function delta(x::T, σ::T)::T where T<:Float64
    σ_inv = INV_SQRT_2π / σ
    exp_factor = -0.5 / (σ^2)
    return σ_inv * exp(x^2 * exp_factor)
end

@inline function FastConv1D(f::AbstractVector{T}, g::AbstractVector{T})::Vector{Complex{T}} where T<:Float64
    return ifft(fft(f).*fft(g))
end


@inline function FastLinearConvolution(f::AbstractVector{T}, g::AbstractVector{T}, power_2_length::Int64)::Vector{Complex{T}} where T<:Float64
    pad_and_ensure_power_of_two!(f, g, power_2_length)
    return FastConv1D(f, g)
end


function is_power_of_two(n::Int64)::Bool
    return (n & (n - 1)) == 0 && n > 0
end


function next_power_of_two(n::Int64)::Int64
    return Int64(2^(ceil(log2(n))))
end



function create_simulation_buffers(n_particles::Int64, nbins::Int64, T::Type=Float64)
    # Pre-allocate all vectors in parallel groups based on size
    particle_vectors = Vector{Vector{T}}(undef, 9)  # For n_particles sized vectors
    bin_vectors = Vector{Vector{T}}(undef, 2)      # For nbins sized vectors
    
    # Initialize n_particles sized vectors in parallel
    Threads.@threads for i in 1:9
        particle_vectors[i] = Vector{T}(undef, n_particles)
    end
    
    # Initialize nbins sized vectors in parallel
    Threads.@threads for i in 1:2
        bin_vectors[i] = Vector{T}(undef, nbins)
    end
    
    # Complex vector (single allocation)
    complex_vector = Vector{Complex{T}}(undef, nbins)
    
    SimulationBuffers{T}(
        particle_vectors[1],  # WF
        particle_vectors[2],  # potential
        particle_vectors[3],  # Δγ
        particle_vectors[4],  # η
        particle_vectors[5],  # coeff
        particle_vectors[6],  # temp_z
        particle_vectors[7],  # temp_ΔE
        particle_vectors[8],  # temp_ϕ
        bin_vectors[1],      # WF_temp
        bin_vectors[2],      # λ
        complex_vector,       # convol
        particle_vectors[9]     #ϕ
    )
end



function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T<:Float64
    N::Int64 = length(f)
    M::Int64 = length(g)
    
    original_f = copy(f)
    resize!(f, power_two_length)
    f[1:N] = original_f
    f[N+1:end] .= zero(T)
    
    original_g = copy(g)
    resize!(g, power_two_length)
    g[1:M] = original_g
    g[M+1:end] .= zero(T)
    
    return nothing
end


function fast_reset_buffers!(buffers::SimulationBuffers{T}) where T<:Float64
    @turbo for i in eachindex(buffers.WF)
        buffers.WF[i] = zero(T)
        buffers.potential[i] = zero(T)
        buffers.Δγ[i] = zero(T)
        buffers.η[i] = zero(T)
        buffers.coeff[i] = zero(T)
        buffers.temp_z[i] = zero(T)
        buffers.temp_ΔE[i] = zero(T)
        buffers.temp_ϕ[i] = zero(T)
    end

    @turbo for i in eachindex(buffers.WF_temp)
        buffers.WF_temp[i] = zero(T)
        buffers.λ[i] = zero(T)
        buffers.convol[i] = zero(Complex{T})
    end

    return nothing
end

function reset_specific_buffers!(buffers::SimulationBuffers{T}, buffer_names::Vector{Symbol}) where T<:Float64
    for name in buffer_names
        buffer = getfield(buffers, name)
        if !all(iszero, buffer)
            fill!(buffer, isa(eltype(buffer), Complex) ? zero(Complex{T}) : zero(T))
        end
    end
    return nothing
end

@inline function calculate_histogram(data::Vector{Float64}, bins_edges)
    histo = Hist1D(data, binedges=bins_edges)
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end


@inline function z_to_ϕ(z_vals::Vector{T}, rf_factor::T, ϕs::T) where T<:Float64
    return -(z_vals * rf_factor .- ϕs)
end

@inline function ϕ_to_z(ϕ_vals::Vector{T}, rf_factor::T, ϕs::T, n_particles::Int64, particles::StructVector{@NamedTuple{coordinates::Coordinate{Float64}, uncertainty::Coordinate{Float64}, derivative::Coordinate{Float64}, derivative_uncertainty::Coordinate{Float64}}, @NamedTuple{coordinates::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}, derivative_uncertainty::StructArrays.StructVector{Coordinate{Float64}, @NamedTuple{z::Vector{Float64}, ΔE::Vector{Float64}}, Int64}}, Int64}#::StructArray{Particle{T}}
    ) where T<:Float64
    @turbo for i in 1:n_particles
        particles.coordinates.z[i] = (-ϕ_vals[i] + ϕs) / rf_factor
    end
end

@inline function z_to_ϕ(z_val::T, rf_factor::T, ϕs::T) where T<:Float64
    return -(z_val * rf_factor - ϕs)
end

@inline function calc_rf_factor(freq_rf::T, β0::T) where T<:Float64
    return freq_rf * 2π / (β0 * SPEED_LIGHT)
end
