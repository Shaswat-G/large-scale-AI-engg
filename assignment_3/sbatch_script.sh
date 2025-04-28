#!/bin/bash
#SBATCH --job-name=model_test  # A name for your job. Visible in squeue.
#SBATCH --account=a-large-sc
#SBATCH --nodes=1 # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1 # Tasks (processes) per node
#SBATCH --time=00:10:00 # HH:MM:SS, set a time limit for this job (here 10min)
#SBATCH --partition=debug # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288 # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --environment=my_env # the environment to use
#SBATCH --output=/iopsstor/scratch/cscs/%u/my_first_sbatch.out # log file for stdout / prints etc
#SBATCH --error=/iopsstor/scratch/cscs/%u/my_first_sbatch.out # log file for stderr / errors


# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail
# Print SLURM variables so you see how your resources are allocated
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Node(s): $SLURM_NODELIST"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Current path: $(pwd)"
echo "Current user: $(whoami)"


source /users/shagupta/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/shagupta/scratch/assignment-2
python3 test_model.py