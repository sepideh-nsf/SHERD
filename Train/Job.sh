#!/usr/bin/env bash
#SBATCH --job-name=jobname
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=600:05:00               # Time limit hrs:min:sec
#SBATCH --output=out.txt         # Standard output and error log
#SBATCH --partition=haswell           # MOAB/Torque called these queues
#SBATCH --mem=16G

python -u Our_method_train.py