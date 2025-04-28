#!/bin/bash --login
#SBATCH --job-name=mpi_gpu_test
#SBATCH --output=slurm/out/test.out
#SBATCH --error=slurm/err/test.err
#SBATCH --time=00:01:59
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=h200:1     
#SBATCH --mem-per-gpu=32G 
#SBATCH --constraint=amd24

# Set critical CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export JULIA_CUDA_USE_BINARYBUILDER=false  # Use system CUDA
export JULIA_CUDA_MEMORY_POOL=none  # Disable memory pooling
export UCX_TLS=sm,tcp,self  # Set UCX transport layers (avoid CUDA-IPC)
export UCX_NET_DEVICES=eth0  # Specify network device


module purge
module load Julia/1.9.3-linux-x86_64
module load likwid
module load OpenMPI

# Print diagnostics
nvidia-smi

mpiexecjl -n 2 julia --project=. tests/benchmark_total.jl --mpi --gpu --turns 1000 --particles 100000

# mpiexecjl --bind-to none -n 2 julia --project=. tests/benchmark_total.jl --mpi --gpu --turns 1000 --particles 100000



# ./clean_cuda_env.sh tests/benchmark_total.jl --gpu --turns 100 --particles 10000


# julia --project=. tests/gpu_test.jl