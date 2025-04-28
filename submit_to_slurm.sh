#!/bin/bash
# Create directory structure
mkdir -p slurm/err slurm/out slurm/scripts

# Define maximum cores and parameter combinations
MAX_CORES=512
PROCESSES=(1 2 4 8 16 32 64 128 256 512)  # Number of MPI processes
THREADS=(1 2 4 8 16 32 64 128 256 512)    # Number of threads per process
TURNS=(1e2 1e3 1e4)
PARTICLES=(1e5 1e6 1e7)

# PROCESSES=(1)  # Number of MPI processes
# THREADS=(1)    # Number of threads per process
# TURNS=(1e1)
# PARTICLES=(1e5)


# Create log directories if they don't exist
mkdir -p logs/mpi_run logs/serial_run

# Generate SLURM job files and submit them
for proc in "${PROCESSES[@]}"; do
  for thread in "${THREADS[@]}"; do
    # Skip combinations that exceed MAX_CORES
    total_cores=$((proc * thread))
    if [ $total_cores -gt $MAX_CORES ]; then
      continue
    fi
    
    for turn in "${TURNS[@]}"; do
      for particle in "${PARTICLES[@]}"; do
        # Create unique job name
        job_name="p${proc}_t${thread}_n${turn}_p${particle}"
        script_file="slurm/scripts/${job_name}.sh"
        
        # Create SLURM script
        cat > "$script_file" << EOL
#!/bin/bash --login
#SBATCH --job-name=${job_name}
#SBATCH --output=slurm/out/${job_name}.out
#SBATCH --error=slurm/err/${job_name}.err
#SBATCH --ntasks=${proc}
#SBATCH --cpus-per-task=${thread}
#SBATCH --time=03:59:59
#SBATCH --mem=493G
#SBATCH --constraint=amd20

module purge
# Load any necessary modules here
module load Julia/1.9.3-linux-x86_64
module load likwid
module load OpenMPI


# Change to directory with code (adjust path as needed)
cd \$SLURM_SUBMIT_DIR

# Run benchmark with MPI
mpiexecjl -n ${proc} julia -t ${thread} --project=. tests/benchmark_total.jl --mpi --turns ${turn} --particles ${particle}

# Note: The benchmark_total.jl script already handles logging based on its parameters
EOL

        # Make script executable
        chmod +x "$script_file"
        
        # Submit job
        echo "Submitting job: $job_name"
        sbatch "$script_file" &
      done
    done
  done
done

echo "All jobs submitted!"