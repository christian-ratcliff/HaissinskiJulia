# StochasticHaissinski

A high-performance beam evolution simulation for particle accelerators with MPI parallelization and stochastic parameter sensitivity analysis.

## Overview

StochasticHaissinski implements a longitudinal beam dynamics simulation in accelerator physics, supporting both serial execution and MPI-based parallelization. The code simulates particle beams through multiple turns in an accelerator, including effects such as:

- RF cavity interactions
- Synchrotron radiation damping
- Quantum excitation
- Collective effects (wakefields)
- Energy-dependent slip factors

The simulation is optimized for performance using various Julia acceleration techniques and is validated for correctness across different MPI configurations.

## Installation

On the HPCC, you must

```bash
module purge
module load Julia
module load OpenMPI
module load likwid
module load powertools
module load matplotlib
julia --project=. -e 'using Pkg; Pkg.instantiate();'
```

## Running Verification Tests
The verification tests ensure that the MPI parallelization produces consistent results regardless of the number of ranks used.
Basic Verification
Run the basic verification script with:
```bash
# Make the script executable
chmod +x run_verifications.sh

# Run with default parameters
./run_verifications.sh

# Run with custom parameters
./run_verifications.sh --turns 200 --particles 20000 --ranks "1,2,4,8"
```

Because of the inherently stochastic nature of the process, there will always be a bit of variation in the stable point, which is why I have set my tolerance so large. 

## Verification Options

--turns N: Set the number of simulation turns (default: 100)
--particles N: Set the number of particles (default: 10000)
--ranks "list": Comma-separated list of MPI ranks to test (default: "1,2,4")
--tolerance N: Set the relative tolerance for comparison (default: 1e-10)

## Understanding Verification Results
The verification script compares simulations run with different numbers of MPI ranks against a reference run with 1 MPI rank. For each configuration, it reports:

The final energy spread (σ_E)
The final bunch length (σ_z)
The final reference energy (E0)
Relative differences between multi-rank and single-rank runs
Overall PASS/FAIL status based on the tolerance

A successful verification will show "Overall Verification Suite Result: PASS" in the summary.

## Running Benchmark Analysis
The code includes performance benchmarking for different loop vectorization strategies.
### Running the Benchmarks
The benchmark plotted by the Python script compares different loop implementation methods:

Serial (standard Julia loop)
@turbo (LoopVectorization.jl)
@floop (FLoops.jl)
ThreadsX (ThreadsX.jl)


```bash
./tests/loops_benchmarks.sh
```

This will run all of the benchmarks and make plots for

Median execution time vs. number of threads
Memory usage vs. number of threads

### Output
The plots are saved to logs/benchmarks/loop_types_benchmarks/<kernel_name>/.


## Running Full Simulations

You can run a single simulation using

```bash
mpiexecjl -n <NUMBER_OF_RANKS> julia benchmark_total.jl --turns <NUMBER_OF_TURNS> --particles <NUMBER_OF_PARTICLES> --mpi
```

or you can choose to run the entire battery of simulations using 
```bash
./run_scaling_studies.sh
```
and then 
```bash
./run_analysis
```
for the log files to be parsed and extracted into a csv, and the plotted. 