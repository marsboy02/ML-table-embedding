# train_multi_ddp.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
import pandas as pd

# -------------------------------
# modeling.py에 있는 것들이라고 가정
# 실제로는 아래 import에서 에러가 없는지 확인 필요
# -------------------------------
from modeling import (MyTableDataset,
                      VerticalSelfAttention,
                      TableCrossEncoder,
                      EarlyStopping,
                      collate_fn_with_row_padding,
                      evaluate)

def main():
    # -------------------------------
    # 1) 프로세스 그룹 초기화
    # -------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun이 자동 세팅
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # -------------------------------
    # 2) 데이터 준비
    # -------------------------------
    print(f"[Rank {dist.get_rank()}] Loading CSV data ...")
    df = pd.read_csv('./final_df_log.csv')

    df['IVS'] = 0.91 * df['Schema_sim_with_col_datas_scaled'] + 0.09 * df['Scaled_Diversity']
    tabledata = df[['Query', 'Target', 'IVS']].dropna()

    if dist.get_rank() == 0:
        print(f'총 학습 데이터 개수 : {tabledata.shape[0]}')

    train_df, val_df = train_test_split(tabledata, test_size=0.3, random_state=42)

    if dist.get_rank() == 0:
        print(f'train dataset : {train_df.shape[0]}')
        print(f'validation dataset : {val_df.shape[0]}')

    # -------------------------------
    # 3) Dataset & Sampler & DataLoader
    # -------------------------------
    train_dataset = MyTableDataset(train_df, input_table_folder='input_table')
    val_dataset   = MyTableDataset(val_df,   input_table_folder='input_table')

    # 멀티 노드/멀티 GPU에서, 각 프로세스마다 부분 데이터를 로드하도록 DistributedSampler 사용
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    # DataLoader: batch_size나 num_workers 적절히 조절
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        collate_fn=collate_fn_with_row_padding,
        num_workers=4,
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

    # -------------------------------
    # 4) 모델 생성 & DDP 래핑
    # -------------------------------
    vertical_attn = VerticalSelfAttention(embed_dim=256, num_heads=4, rep_mode="cls").to(device)
    cross_encoder = TableCrossEncoder(pretrained_model_name="bert-base-uncased", hidden_dim=256).to(device)

    # DDP로 감싸기
    vertical_attn = DDP(vertical_attn, device_ids=[local_rank],
                        output_device=local_rank, find_unused_parameters=True)
    cross_encoder = DDP(cross_encoder, device_ids=[local_rank],
                        output_device=local_rank, find_unused_parameters=True)

    # -------------------------------
    # 5) Optimizer 및 EarlyStopping 설정
    # -------------------------------
    params = list(vertical_attn.parameters()) + list(cross_encoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    early_stopping = EarlyStopping(patience=5, delta=0.0,
                                   save_path='best_model_sche9_div1.pt')

    # -------------------------------
    # 6) 학습 루프
    # -------------------------------
    num_epochs = 50
    for epoch in range(num_epochs):
        # DDP에서 epoch마다 sampler의 랜덤 시드 설정(셔플 동기화)
        train_sampler.set_epoch(epoch)

        vertical_attn.train()
        cross_encoder.train()

        total_loss = 0.0
        for batch_idx, (table1s, table2s, scores, real_cols1, real_cols2, real_rows1, real_rows2) in enumerate(train_loader):
            table1s = table1s.to(device)
            table2s = table2s.to(device)
            scores  = scores.to(device)
            real_cols1 = real_cols1.to(device)
            real_cols2 = real_cols2.to(device)

            table1_reps = vertical_attn(table1s, real_cols1, real_rows1)
            table2_reps = vertical_attn(table2s, real_cols2, real_rows2)

            pred_scores = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)

            loss = F.mse_loss(pred_scores, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # (옵션) 모든 rank에서 계산된 total_loss의 평균을 구해보고 싶다면, 아래처럼 통신 연산 사용 가능
        # 여기서는 간단히 rank=0에 대해서만 로그 남기는 정도로 처리.
        
        avg_train_loss = total_loss / len(train_loader)

        # Validation은 모든 rank에서 수행해도 되지만,
        # 보통 rank=0(마스터)에서만 로깅/모델저장 처리를 합니다.
        if dist.get_rank() == 0:
            val_loss = evaluate(vertical_attn.module, cross_encoder.module, val_loader, device)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")

            model_dict = {
                'vertical_attn': vertical_attn.module.state_dict(),
                'cross_encoder': cross_encoder.module.state_dict()
            }
            early_stopping(val_loss, model_dict)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # -------------------------------
    # 7) 모든 작업 완료 후 프로세스 그룹 해제
    # -------------------------------
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
