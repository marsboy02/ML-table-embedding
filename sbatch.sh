#!/bin/bash
#SBATCH --nodes=5
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ysh1116
#SBATCH -o ./results/jupyter.%N.%j.out  # STDOUT
#SBATCH -e ./results_log/jupyter.%N.%j.err  # STDERR

echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

# set slurm module
module unload CUDA/11.2.2
module load CUDA/11.2.2
module add python/3.11.2

# set envrionment
python3 -m venv venv
source venv/bin/activate

# install dependency
pip3 install torch sklearn.model_selection modeling pandas transformers scikit-learn numpy

python3 train.py