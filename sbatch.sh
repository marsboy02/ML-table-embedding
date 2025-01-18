#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test_root
#SBATCH -o ./results/jupyter.%N.%j.out  # STDOUT
#SBATCH -e ./results_log/jupyter.%N.%j.err  # STDERR

echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

# set slurm module
module load CUDA/11.2.2
module add python/3.11.2

# set envrionment
python3 -m venv venv
source venv/bin/activate

# install dependency
pip3 install pandas
pip3 install torch
pip3 install scikit-learn
pip install --upgrade pip setuptools
pip3 install transformers
pip3 install numpy

python3 train.py