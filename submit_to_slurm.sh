#!/bin/bash

# Create directory structure
mkdir -p slurm/err slurm/out slurm/scripts

# Define parameter combinations
THREADS=(1 2 4 8 16 32 64 96 128 192)
TURNS=(1e2 1e3 1e4)
PARTICLES=(1e5 1e6 1e7)


# Create log directories if they don't exist
mkdir -p logs/initial

# Generate SLURM job files and submit them
for thread in "${THREADS[@]}"; do
    for turn in "${TURNS[@]}"; do
        for particle in "${PARTICLES[@]}"; do
            # Create unique job name
            job_name="t${thread}_n${turn}_p${particle}"
            script_file="slurm/scripts/${job_name}.sh"
            
            # Create SLURM script
            cat > "$script_file" << EOL
#!/bin/bash --login
#SBATCH --job-name=${job_name}
#SBATCH --output=slurm/out/${job_name}.out
#SBATCH --error=slurm/err/${job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${thread}
#SBATCH --time=02:00:00
#SBATCH --mem=493G
#SBATCH --constraint=amd24

module purge 

# Load any necessary modules here
module load Julia
module load likwid

# Change to directory with code (adjust path as needed)
cd \$SLURM_SUBMIT_DIR

# Pre-process the Julia script to set n_turns and n_particles
tmp_script="tests/tmp_script_${job_name}.jl"
cp tests/benchmarks.jl \$tmp_script

# Replace n_turns and n_particles with the specific values
sed -i "s/n_turns = [0-9]*\|n_turns = [0-9.]\+e[0-9]\+/n_turns = ${turn}/g" \$tmp_script
sed -i "s/n_particles = Int64([0-9.]\+e[0-9]\+)/n_particles = Int64(${particle})/g" \$tmp_script

# Run Julia with the specified number of threads
julia --project=. -t ${thread} \$tmp_script

# Clean up temporary script
rm \$tmp_script
EOL

            # Make script executable
            chmod +x "$script_file"
            
            # Submit job
            echo "Submitting job: $job_name"
            sbatch "$script_file" &
        done
    done
done

echo "All jobs submitted!"