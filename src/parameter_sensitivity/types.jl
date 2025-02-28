"""
types.jl - Type definitions for parameter sensitivity analysis

This file defines the abstract and concrete types used in parameter sensitivity analysis:
- ParameterTransformation: Abstract type for parameter transformations
- FigureOfMerit: Abstract type for beam quality metrics
- ParameterSensitivity: Container for sensitivity analysis results
"""

"""
    ParameterTransformation

Abstract type for all parameter transformations.
"""
abstract type ParameterTransformation end

"""
    VoltageTransform <: ParameterTransformation

Transformation for RF voltage parameter.
"""
struct VoltageTransform <: ParameterTransformation end

"""
    AlphaCompactionTransform <: ParameterTransformation

Transformation for momentum compaction parameter.
"""
struct AlphaCompactionTransform <: ParameterTransformation end

"""
    HarmonicNumberTransform <: ParameterTransformation

Transformation for harmonic number parameter.
"""
struct HarmonicNumberTransform <: ParameterTransformation end

"""
    PipeRadiusTransform <: ParameterTransformation

Transformation for beam pipe radius parameter.
"""
struct PipeRadiusTransform <: ParameterTransformation end

"""
    FigureOfMerit

Abstract type for all figures of merit.
"""
abstract type FigureOfMerit end

"""
    EnergySpreadFoM <: FigureOfMerit

Figure of merit: Energy spread.
"""
struct EnergySpreadFoM <: FigureOfMerit end

"""
    BunchLengthFoM <: FigureOfMerit

Figure of merit: Bunch length.
"""
struct BunchLengthFoM <: FigureOfMerit end

"""
    EmittanceFoM <: FigureOfMerit

Figure of merit: Emittance.
"""
struct EmittanceFoM <: FigureOfMerit end

"""
    ParameterSensitivity

Structure containing sensitivity analysis for a parameter.
"""
struct ParameterSensitivity
    transform::ParameterTransformation
    fom::FigureOfMerit
    param_value::Float64
    mean_derivative::Float64
    uncertainty::Float64
    samples::Vector{Float64}
end