"""
StochasticHaissinski.jl - Main module for stochastic Haissinski simulations

This module implements a high-performance beam evolution simulation for particle accelerators
with StochasticAD integration for parameter sensitivity analysis.
"""
module StochasticHaissinski

# Standard library imports
using Statistics
using LinearAlgebra
using Random

# External dependencies
using StochasticAD
using Distributions
using StructArrays
using LoopVectorization
using FFTW
using Interpolations
using ProgressMeter
using FHist
# using Plots
using LaTeXStrings

try
    using CUDA
catch e
    @warn "CUDA.jl not available: $e"
end

# Include submodules
include("constants.jl")
include("data_structures.jl")
include("utils.jl")
include("rf_kick.jl")
include("synchrotron_radiation.jl")
include("evolution.jl")
include("quantum_excitation.jl")
include("wakefield.jl")
include("visualization.jl")

# Include GPU implementations if CUDA is available
const has_cuda = isfile(joinpath(@__DIR__, "gpu_data_structures.jl")) && 
                (@isdefined CUDA) && isdefined(CUDA, :functional) && CUDA.functional()

if has_cuda
    include("gpu_data_transfer.jl")
    include("gpu_utils.jl")
    include("gpu_rf_kick.jl")
    include("gpu_synchrotron_radiation.jl")
    include("gpu_quantum_excitation.jl")
    include("gpu_phase_advance.jl")
    include("gpu_wakefield.jl")
end

# Include parameter sensitivity framework
include("parameter_sensitivity/types.jl")
include("parameter_sensitivity/transformations.jl")
include("parameter_sensitivity/figures_of_merit.jl")
include("parameter_sensitivity/sensitivity.jl")
include("parameter_sensitivity/scan.jl")
include("parameter_sensitivity/visualization.jl")

# Export constants
export SPEED_LIGHT, ELECTRON_CHARGE, MASS_ELECTRON

# Export data structures
export Coordinate, Particle, BeamTurn, SimulationParameters, SimulationBuffers

# Export core functions
export generate_particles, longitudinal_evolve!
export quantum_excitation!, synchrotron_radiation!, apply_wakefield_inplace!, rf_kick!, synchrotron_radiation!

# Export utilities
export z_to_ϕ, ϕ_to_z, calc_rf_factor, create_simulation_buffers,copyto_particles!, next_power_of_two, initialize_mpi_buffers!, free_mpi_buffer!, precompute_thread_chunks!

# Export parameter sensitivity
export VoltageTransform, AlphaCompactionTransform, HarmonicNumberTransform, PipeRadiusTransform, ParameterTransformation, apply_transform, compute_fom, ParameterSensitivity
export EnergySpreadFoM, BunchLengthFoM, EmittanceFoM, FigureOfMerit
export compute_sensitivity, scan_parameter, plot_sensitivity_scan

# Export GPU-related functionality
if has_cuda
    export is_gpu_available, initialize_gpu, GPUConfig
    export GPUParticleData, GPUSimulationBuffers
    export transfer_particles_to_gpu!, transfer_particles_to_cpu!
    export is_gpu_aware_mpi_available
end

end # module