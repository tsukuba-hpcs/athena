#!/bin/bash
#SBATCH --job-name=IO
#SBATCH --partition=M-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:10:00

cd ${SLURM_SUBMIT_DIR}
srun ./athena -i athinput.iobench_smr512mb  > logsmr512mb
