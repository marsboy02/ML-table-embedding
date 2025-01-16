#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=56
#SBATCH --job-name=test_root
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
pip3 install torch
pip3 install sklearn.model_selection
pip3 install modeling
pip3 install pandas
pip3 install transformers
pip3 install scikit-learn
pip3 install numpy

python3 train.py