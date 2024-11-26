#!/bin/bash
#
#SBATCH --job-name=LeonJupyter # Job name for tracking
#SBATCH --partition=eagle      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err
#SBATCH --mail-type=BEGIN # Get email when your allocation starts

source /etc/profile.d/modules.sh
module load CUDA
module load python-ml
cd ../src
jupyter notebook --ip=0.0.0.0 --port=11862 --no-browser