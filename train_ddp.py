# train_ddp.py

import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

# modeling.py에 정의된 클래스/함수들을 import 해옵니다.
from modeling import MyTableDataset, VerticalSelfAttention, TableCrossEncoder, EarlyStopping
from modeling import collate_fn_with_row_padding, evaluate

def main():
    """
    Slurm 스크립트 (예: sbatch_ddp.sh)에서
    torchrun --nproc_per_node=4 train_ddp.py
    형태로 실행했을 때 4개의 프로세스가 동시 실행됩니다.
    """

    # ---------------------------------------------
    # 1) 프로세스 그룹 초기화
    # ---------------------------------------------
    dist.init_process_group(backend="nccl") 
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun이 자동으로 넘겨주는 local_rank
    torch.cuda.set_device(local_rank)           # 각 프로세스는 자신에게 할당된 GPU만 사용
    device = torch.device("cuda", local_rank)

    # ---------------------------------------------
    # 2) 데이터 준비
    # ---------------------------------------------
    print(f"[Rank {dist.get_rank()}] Loading data ...")
    df = pd.read_csv('./final_df_log.csv')

    # 아래는 기존 코드와 동일
    df['IVS'] = 0.91 * df['Schema_sim_with_col_datas_scaled'] + 0.09 * df['Scaled_Diversity']
    tabledata = df[['Query', 'Target', 'IVS']].dropna()
    
    print(f'총 학습 데이터 개수 : {tabledata.shape[0]}')
    train_df, val_df = train_test_split(tabledata, test_size=0.3, random_state=42)
    print(f'train dataset : {train_df.shape[0]}')
    print(f'validation dataset : {val_df.shape[0]}')

    # ---------------------------------------------
    # 3) Dataset & DistributedSampler & DataLoader
    # ---------------------------------------------
    train_dataset = MyTableDataset(train_df, input_table_folder='input_table')
    val_dataset   = MyTableDataset(val_df,   input_table_folder='input_table')

    # DDP에서 각 프로세스가 데이터를 나누어 받도록 분산 샘플러 사용
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    # DataLoader - batch_size와 collate_fn은 기존과 동일
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        collate_fn=collate_fn_with_row_padding,
        num_workers=4,  # CPU 코어 상황에 맞춰 조정
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        sampler=val_sampler,
        collate_fn=collate_fn_with_row_padding,
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------------------------
    # 4) 모델 생성 및 DDP 래핑
    # ---------------------------------------------
    vertical_attn = VerticalSelfAttention(embed_dim=256, num_heads=4, rep_mode="cls").to(device)
    cross_encoder = TableCrossEncoder(pretrained_model_name="bert-base-uncased", hidden_dim=256).to(device)

    # DistributedDataParallel로 감싸기
    # find_unused_parameters=True는 어떤 branch가 분기되어 사용되지 않는 파라미터가 있을 때를 대비.
    vertical_attn = DDP(vertical_attn, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    cross_encoder = DDP(cross_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ---------------------------------------------
    # 5) Optimizer 및 EarlyStopping 설정
    # ---------------------------------------------
    params = list(vertical_attn.parameters()) + list(cross_encoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    early_stopping = EarlyStopping(patience=5, delta=0.0, save_path='best_model_sche9_div1.pt')

    # ---------------------------------------------
    # 6) 학습 루프
    # ---------------------------------------------
    num_epochs = 50
    for epoch in range(num_epochs):
        # DDP에서 epoch마다 sampler 시드를 고정해야 제대로 shuffle 됨
        train_sampler.set_epoch(epoch)

        vertical_attn.train()
        cross_encoder.train()
        
        total_loss = 0.0
        for batch_idx, (table1s, table2s, scores, real_cols1, real_cols2, real_rows1, real_rows2) in enumerate(train_loader):
            # GPU로 텐서 복사
            table1s = table1s.to(device)
            table2s = table2s.to(device)
            scores  = scores.to(device)
            real_cols1 = real_cols1.to(device)
            real_cols2 = real_cols2.to(device)

            # Forward
            table1_reps = vertical_attn(table1s, real_cols1, real_rows1)  
            table2_reps = vertical_attn(table2s, real_cols2, real_rows2)  

            # Cross Encoder(BERT)
            pred_scores = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)

            # Loss
            loss = F.mse_loss(pred_scores, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---------------------------------------------
        # 7) (옵션) Validation & EarlyStopping
        #    -> 모든 프로세스에서 해도 되지만,
        #       보통 rank=0(마스터)에서만 결과를 출력하고
        #       모델을 저장합니다.
        # ---------------------------------------------
        avg_train_loss = total_loss / len(train_loader)
        
        if dist.get_rank() == 0:
            # evaluate 함수가 내부적으로 val_loader를 순회하며 loss를 측정
            # DDP 래핑된 모델에서는 'model.module' 형태로 실제 모델 접근
            val_loss = evaluate(vertical_attn.module, cross_encoder.module, val_loader, device)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # EarlyStopping 체크
            model_dict = {
                'vertical_attn':   vertical_attn.module.state_dict(),
                'cross_encoder':   cross_encoder.module.state_dict()
            }
            early_stopping(val_loss, model_dict)

            if early_stopping.early_stop:
                print("Early stopping is triggered!")
                break

    # ---------------------------------------------
    # 8) 프로세스 그룹 종료
    # ---------------------------------------------
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
