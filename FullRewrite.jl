begin
    include("src/StochasticHaissinski.jl")
end


begin
    using .StochasticHaissinski
    using BenchmarkTools
    using CairoMakie
    using StructArrays
    using StochasticAD
end

begin
    # Physical constants
    const SPEED_LIGHT::Float64 = 299792458.
    const ELECTRON_CHARGE::Float64 = 1.602176634e-19
    const MASS_ELECTRON::Float64 = 0.51099895069e6
    const INV_SQRT_2π::Float64 = 1 / sqrt(2 * π)
    const ħ::Float64 = 6.582119569e-16
end;

begin
    E0_ini::Float64 = 4e9
    mass::Float64 = MASS_ELECTRON
    voltage::Float64 = 5e6
    harmonic::Int64 = 360
    radius::Float64 = 250.
    pipe_radius::Float64 = .00025
    α_c::Float64 = 3.68e-4
    γ::Float64 = E0_ini/mass
    β::Float64 = sqrt(1 - 1/γ^2)
    η::Float64= α_c - 1/γ^2
    sin_ϕs::Float64 = 0.5
    ϕs::Float64 = 5π/6
    freq_rf::Float64 = (ϕs + 10 *π/180) * β * SPEED_LIGHT / (2π)
    μ_E::Float64 = 0.
    μ_z::Float64 = 0.
    ω_rev::Float64 = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT))
    σ_E0::Float64 = 1e6
    σ_z0::Float64 = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini
end;

n_turns::Int64 = 100;
particles, σ_E, σ_z, E0 = generate_particles(μ_z, μ_E, σ_z0,σ_E0, Int64(1e2),E0_ini,mass,ϕs, freq_rf) ;

# particles.coordinates.z

σ_E, σ_z, E0= longitudinal_evolve!(
    n_turns, particles, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true);

# longitudinal_evolve!(
#     n_turns, particles, ϕs, α_c, mass, voltage,
#     harmonic, radius, freq_rf, pipe_radius, E0, σ_E,σ_z,
#     use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
#     use_excitation=true)

particles.coordinates.ΔE
particles.derivative.ΔE
particles.derivative_uncertainty.ΔE

begin
    fig = Figure(;size = (800, 500))
    ax = Axis(fig[1,1], xlabel=L"\Delta E / \sigma_E", ylabel="Count", title = "Energy Distribution") 
    hist!(ax, particles.coordinates.ΔE / σ_E, bins=100)
    display(fig)
end
begin
    fig = Figure(;size = (800, 500))
    ax = Axis(fig[1,1], xlabel=L"\Delta z / \sigma_z", ylabel="Count", title = "z Distribution") 
    hist!(ax, particles.coordinates.z / σ_z, bins=100)
    display(fig)
end

begin
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
    Label(fig[1, 1, Top()], "At Convergence: Turn $(length(particles))", fontsize = 20)
    scatter!(ax, 
    z_to_ϕ(particles.coordinates.z, calc_rf_factor(freq_rf, β), ϕs),
    particles.coordinates.ΔE / σ_E,
    markersize = 10, color = :black)
    scatter!(ax, 
        ϕs,
        0,
        markersize = 10, color = :yellow)
    # println(σ_E)
    # lines!(ax, boundary_obs[][1], boundary_obs[][2] / σ_E0, color=:red)
    # xlims!(ax, 0, 3π/2)
    # ylims!(ax, minimum(boundary_points[2]) / σ_E0-5, maximum(boundary_points[2]) / σ_E0+5)
    display(fig)
end

println(typeof(particles))



@btime particles.derivative.ΔE .= 0


n_particles = length(particles)
buffer = zeros(Float64, n_particles)
using Random
using Distributions
randn!(buffer)
X(p, delE::Float64) = delE + 2. * p
function testfunc!(buffer::Vector{Float64}, n_particles::Int64, particles::StructArray{Particle{T}}
) where T

    samples = zeros(Float64, 100)
    randn!(buffer)::Vector{Float64}
    X(p::StochasticTriple, delE::Float64) = delE + 2. * p
    for i in 1:n_particles
        # particles.coordinates.ΔE[i] = particles.coordinates.ΔE[i] + excitation * buffer[i] #This is not actually the potential, merely a random number with the right distribution, I just use the buffer because its already allocated
        
        samples .= [derivative_estimate(p -> X(p, particles.coordinates.ΔE[i]), randn()) for j in 1:100]
        particles.derivative.ΔE[i] = mean(samples)
        particles.coordinates.ΔE[i] = particles.coordinates.ΔE[i] + 2 * buffer[i]
        # println(mean(samples))
        particles.derivative_uncertainty.ΔE[i] = std(samples) / sqrt(1000)
    end
    
end


function testfunc!(buffer::Vector{Float64}, n_particles::Int64, particles::StructArray{Particle{T}}
    ) where T
    
        samples = zeros(Float64, 100)
        randn!(buffer)  # Fill buffer with standard normal noise
    
        X(p::StochasticTriple, delE::Float64) = delE + 2.0 * p  # Function to differentiate
    
        for i in 1:n_particles
            samples .= [derivative_estimate(p -> X(p, particles.coordinates.ΔE[i]), pdf(Normal(), p)) for _ in 1:100]
            particles.derivative.ΔE[i] = mean(samples)
            particles.coordinates.ΔE[i] += 2 * buffer[i]
            particles.derivative_uncertainty.ΔE[i] = std(samples) / sqrt(100)
        end
    end



samples = [Vector{Float64}(undef, 100) for _ in 1:n_particles]  # Preallocate

testfunc!(buffer, n_particles, particles)

particles.derivative.ΔE
particles.derivative_uncertainty.ΔE


@inbounds for i in 1:n_particles
    for j in 1:100
        samples[i][j] = derivative_estimate(p -> X(p,  particles.coordinates.ΔE[i]), randn!(buffer))
    end
end

X(2, buffer[1])

stochastic_triple(p -> X(p, buffer[1]), particles.coordinates.ΔE[1] )
derivative_estimate(p -> X(p, buffer[1]), particles.coordinates.ΔE[1])
samples

