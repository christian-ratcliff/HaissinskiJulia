"""
    apply_transform(transform::VoltageTransform, param_value, sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF})

Apply voltage transformation to simulation parameters, preserving all other parameter types.
"""
function apply_transform(transform::VoltageTransform, param_value, 
                        sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}) where {TE,TM,TV,TR,TPR,TA,TPS,TF}
    # Use the type of param_value for voltage, keep all other types the same
    return SimulationParameters{TE,TM,typeof(param_value),TR,TPR,TA,TPS,TF}(
        sim_params.E0,
        sim_params.mass,
        param_value,  # The parameter being transformed
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
end

"""
    apply_transform(transform::AlphaCompactionTransform, param_value, sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF})

Apply momentum compaction transformation to simulation parameters, preserving all other parameter types.
"""
function apply_transform(transform::AlphaCompactionTransform, param_value, 
                        sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}) where {TE,TM,TV,TR,TPR,TA,TPS,TF}
    # Use the type of param_value for α_c, keep all other types the same
    return SimulationParameters{TE,TM,TV,TR,TPR,typeof(param_value),TPS,TF}(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        sim_params.harmonic,
        sim_params.radius,
        sim_params.pipe_radius,
        param_value,  # The parameter being transformed
        sim_params.ϕs,
        sim_params.freq_rf,
        sim_params.n_turns,
        sim_params.use_wakefield,
        sim_params.update_η,
        sim_params.update_E0,
        sim_params.SR_damping,
        sim_params.use_excitation
    )
end

"""
    apply_transform(transform::PipeRadiusTransform, param_value, sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF})

Apply pipe radius transformation to simulation parameters, preserving all other parameter types.
"""
function apply_transform(transform::PipeRadiusTransform, param_value, 
                        sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}) where {TE,TM,TV,TR,TPR,TA,TPS,TF}
    # Use the type of param_value for pipe_radius, keep all other types the same
    return SimulationParameters{TE,TM,TV,TR,typeof(param_value),TA,TPS,TF}(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        sim_params.harmonic,
        sim_params.radius,
        param_value,  # The parameter being transformed
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
end

"""
    apply_transform(transform::HarmonicNumberTransform, param_value, sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF})

Apply harmonic number transformation to simulation parameters, preserving all other parameter types.
Special handling needed since harmonic is always Int type.
"""
function apply_transform(transform::HarmonicNumberTransform, param_value, 
                        sim_params::SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}) where {TE,TM,TV,TR,TPR,TA,TPS,TF}
    # Since harmonic is always Int, just round and convert the param_value
    return SimulationParameters{TE,TM,TV,TR,TPR,TA,TPS,TF}(
        sim_params.E0,
        sim_params.mass,
        sim_params.voltage,
        Int(round(param_value)),  # The parameter being transformed
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
end