#!/bin/bash

#SBATCH --job-name=ddp_multinode       # 잡 이름
#SBATCH --nodes=2                      # 사용할 노드 수 (예: 2)
#SBATCH --gres=gpu:4                   # 각 노드 당 GPU 4장
#SBATCH --cpus-per-task=16             # 프로세스 당 사용할 CPU 코어 개수
#SBATCH --ntasks-per-node=4            # 노드당 프로세스 수 -> GPU 개수와 동일
#SBATCH --time=02:00:00                # 잡 최대 실행 시간
#SBATCH -o logs/multinode.%N.%j.out    # STDOUT 로그
#SBATCH -e logs/multinode.%N.%j.err    # STDERR 로그

echo "Starting job at: $(date)"
echo "Node list: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# 모듈/환경 설정 (예시)
module load CUDA/11.2.2
module load python/3.11.2

# 가상환경 활성화
source ~/venv/bin/activate

# -----------------------------
# 필요 패키지 설치 (예: requirements.txt가 있다면)
# pip install -r requirements.txt
# pip install torch torchvision torchaudio
# pip install transformers scikit-learn numpy pandas
# pip install modeling
# -----------------------------

# 멀티 노드 DDP 실행 (torchrun)
# --nnodes=2              => 전체 노드 수
# --nproc_per_node=4      => 노드당 프로세스 수 (GPU 4장)
# --node_rank=$SLURM_NODEID => Slurm가 자동으로 0, 1, ... 부여
# --master_addr=... --master_port=...  => 첫 번째 노드의 주소/포트
MASTER_ADDR=$(srun hostname | head -n1)
MASTER_PORT=12345

torchrun --nnodes=2 \
         --nproc_per_node=4 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train_multi_ddp.py

echo "Job finished at: $(date)"
