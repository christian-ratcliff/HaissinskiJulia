module StochasticHaissinski
using LoopVectorization

begin
    # Physical constants
    const SPEED_LIGHT::Float64 = 299792458.
    const ELECTRON_CHARGE::Float64 = 1.602176634e-19
    const MASS_ELECTRON::Float64 = 0.51099895069e6
    const INV_SQRT_2π::Float64 = 1 / sqrt(2 * π)
    const ħ::Float64 = 6.582119569e-16
end;

export SPEED_LIGHT, ELECTRON_CHARGE, MASS_ELECTRON, INV_SQRT_2π, ħ

include("data_structures.jl")
export Coordinate, Particle, BeamTurn, SimulationBuffers

include("utils.jl")
export threaded_fieldwise_copy!, assign_to_turn!, delta, FastConv1D, FastLinearConvolution, is_power_of_two, next_power_of_two, create_simulation_buffers, pad_and_ensure_power_of_two!, fast_reset_buffers!, reset_specific_buffers!, calculate_histogram,z_to_ϕ, calc_rf_factor, ϕ_to_z

include("evolution.jl")
export generate_particles, BeamTurn, longitudinal_evolve!




end