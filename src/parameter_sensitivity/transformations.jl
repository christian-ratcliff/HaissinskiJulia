"""
transformations.jl - Parameter transformation implementations

This file implements the specific parameter transformations for sensitivity analysis,
defining how each parameter affects the simulation parameters.
"""

"""
    apply_transform(transform::VoltageTransform, param_value, sim_params)

Apply voltage transformation to simulation parameters.
"""
function apply_transform(transform::VoltageTransform, param_value, sim_params)
    # new_params = deepcopy(sim_params)
    new_params = SimulationParameters(
        sim_params.E0,
        sim_params.mass,
        param_value,
        sim_params.harmonic,
        sim_params.radius,
        sim_params.pipe_radius,
        sim_params.α_c,
        sim_params.ϕs,
        sim_params.freq_rf,
        sim_params.n_turns,
        sim_params.use_wakefield,
        sim_params.update_η,
        sim_params.update_E0,
        sim_params.SR_damping,
        sim_params.use_excitation
    )
    return new_params
end

"""
    apply_transform(transform::AlphaCompactionTransform, param_value, sim_params)

Apply momentum compaction transformation to simulation parameters.
"""
function apply_transform(transform::AlphaCompactionTransform, param_value, sim_params)
    # new_params = deepcopy(sim_params)
    # new_params.α_c = param_value
    new_params = SimulationParameters(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        sim_params.harmonic,
        sim_params.radius,
        sim_params.pipe_radius,
        param_value,
        sim_params.ϕs,
        sim_params.freq_rf,
        sim_params.n_turns,
        sim_params.use_wakefield,
        sim_params.update_η,
        sim_params.update_E0,
        sim_params.SR_damping,
        sim_params.use_excitation
    )
    return new_params
end

"""
    apply_transform(transform::HarmonicNumberTransform, param_value, sim_params)

Apply harmonic number transformation to simulation parameters.
"""
function apply_transform(transform::HarmonicNumberTransform, param_value, sim_params)
    # new_params = deepcopy(sim_params)
    # new_params.harmonic = Int(round(param_value))
    new_params = SimulationParameters(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        Int(round(param_value)),
        sim_params.radius,
        sim_params.pipe_radius,
        sim_params.α_c,
        sim_params.ϕs,
        sim_params.freq_rf,
        sim_params.n_turns,
        sim_params.use_wakefield,
        sim_params.update_η,
        sim_params.update_E0,
        sim_params.SR_damping,
        sim_params.use_excitation
    )
    return new_params
end

"""
    apply_transform(transform::PipeRadiusTransform, param_value, sim_params)

Apply pipe radius transformation to simulation parameters.
"""
function apply_transform(transform::PipeRadiusTransform, param_value, sim_params)
    # new_params = deepcopy(sim_params)
    # new_params.pipe_radius = param_value
    new_params = SimulationParameters(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        sim_params.harmonic,
        sim_params.radius,
        param_value,
        sim_params.α_c,
        sim_params.ϕs,
        sim_params.freq_rf,
        sim_params.n_turns,
        sim_params.use_wakefield,
        sim_params.update_η,
        sim_params.update_E0,
        sim_params.SR_damping,
        sim_params.use_excitation
    )
    return new_params
end