#!/bin/bash

module load matplotlib


# julia -t 32 --project=. tests/loop_benchmark.jl
# julia -t 16 --project=. tests/loop_benchmark.jl
# julia -t 8 --project=. tests/loop_benchmark.jl
# julia -t 4 --project=. tests/loop_benchmark.jl
# julia -t 2 --project=. tests/loop_benchmark.jl
# julia -t 1 --project=. tests/loop_benchmark.jl

python tests/process_logs.py

python tests/plot_loop_benchmarks.py