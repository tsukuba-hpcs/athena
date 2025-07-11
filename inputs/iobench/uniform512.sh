#!/bin/bash
#SBATCH --job-name=IO
#SBATCH --partition=M-large-a
#SBATCH --nodes=8
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks-per-socket=32
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:10:00

cd ${SLURM_SUBMIT_DIR}
srun ./athena -i athinput.iobench512p1t  > log512p1t
