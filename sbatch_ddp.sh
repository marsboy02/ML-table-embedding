#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=56
#SBATCH --job-name=test_ddp
#SBATCH -o ./results/ddp.%N.%j.out
#SBATCH -e ./results_log/ddp.%N.%j.err

echo "Start at: " `date`
echo "Node: $HOSTNAME"
echo "JobID: $SLURM_JOB_ID"

module load CUDA/11.2.2
module load python/3.11.2

# 가상환경 세팅
python3 -m venv venv
source venv/bin/activate

# 종속성 설치
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio  # 버전에 맞춰 설치
pip3 install transformers scikit-learn numpy pandas
pip3 install modeling  # modeling.py를 따로 패키지화 했다면
# ↑ 'pip3 insatll' 오타 주의

# DDP 실행
# nproc_per_node=4 -> GPU 4장 활용
torchrun --nproc_per_node=4 train_ddp.py
