#!/bin/bash
#SBATCH --job-name=IO
#SBATCH --partition=M-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks-per-socket=32
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:10:00

cd ${SLURM_SUBMIT_DIR}
srun ./athena -i athinput.iobench64r  > log64r
srun ./athena -r iobench_restart.00001.rst  > log64r2
diff log64r log64r2 > log_diff
