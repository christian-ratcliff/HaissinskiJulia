begin
    using StructArrays, Random, Distributions, LinearAlgebra, StaticArrays
end


struct Coordinate{T} <: FieldVector{2, T}
    z::T
    ΔE::T
end


struct Particle{T} <: FieldVector{4, Coordinate}
    coordinates::Coordinate{T}
    uncertainty::Coordinate{T}
    derivative::Coordinate{T}
    derivative_uncertainty::Coordinate{T}
end

struct BeamTurn{T, N}
    states::Vector{StructArray{Particle{T}}}  # StructArray ensures fast field access
end


struct SimulationBuffers{T<:Float64}
    WF::Vector{T}
    potential::Vector{T}
    Δγ::Vector{T}
    η::Vector{T}
    coeff::Vector{T}
    temp_z::Vector{T}
    temp_ΔE::Vector{T}
    temp_ϕ::Vector{T}
    WF_temp::Vector{T}
    λ::Vector{T}
    convol::Vector{Complex{T}}
    ϕ::Vector{T}
end