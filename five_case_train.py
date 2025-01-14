################################## 1 / 12 ##################################
################################## 1 / 12 ##################################
################################## 1 / 12 ##################################

# scaled_sim, scaled_div를 alpha(0.9, 0.7, 0.5, 0.3, 0.1)
# 인자에 따른 결과 확인 및 중요 정도 비교를 위함

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from modeling import MyTableDataset, VerticalSelfAttention, TableCrossEncoder, EarlyStopping
from modeling import collate_fn_with_row_padding, evaluate

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

print('Dataframe 호출 시작')

# ==================================== <IVS 선택 사항 부분> ====================================
augmented_final_df = pd.read_csv('./final_df.csv')

augmented_final_df['scaled_div'] = scaler.fit_transform(augmented_final_df[['Diversity']])
augmented_final_df['scaled_Schema_sim_with_col_datas'] = scaler.fit_transform(augmented_final_df[['Schema_sim_with_col_datas']])
augmented_final_df['scaled_sim'] = scaler.fit_transform(augmented_final_df[['cosine_sim']])

# refined_df_final을 학습 데이터로 사용

refined_df = augmented_final_df.dropna()

for alpha in [0.3]:

    refined_df['ivs'] = alpha*refined_df['scaled_div'] + (1-alpha)*refined_df['scaled_sim']
    refined_df['IVS'] = scaler.fit_transform(refined_df[['ivs']])

    tabledata = refined_df[['Query', 'Target', 'IVS']]
    tabledata = tabledata.dropna()
    # ============================================================================================

    print(f'총 학습 데이터 개수 : {tabledata.shape[0]}')

    train_df, val_df = train_test_split(tabledata, test_size=0.2, random_state=42)

    print('학습 Dataframe 구축 완료')
    print(f'train dataset : {train_df.shape[0]}')
    print(f'validation dataset : {val_df.shape[0]}')

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device :', device)

    # 하이퍼파라미터
    num_epochs = 50
    batch_size = 8
    lr = 1e-4

    # 1) 데이터셋 / 데이터로더
    train_dataset = MyTableDataset(train_df, input_table_folder='input_table')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_with_row_padding, shuffle=True)

    val_dataset = MyTableDataset(val_df, input_table_folder='input_table')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_with_row_padding, shuffle=True)

    # 2) 모델 초기화
    vertical_attn = VerticalSelfAttention(embed_dim=256, num_heads=4, rep_mode="cls")
    cross_encoder = TableCrossEncoder(pretrained_model_name="distilbert-base-uncased", hidden_dim=256)

    # 2-1) 모델 => device
    vertical_attn.to(device)
    cross_encoder.to(device)

    # 옵티마이저
    params = list(vertical_attn.parameters()) + list(cross_encoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    vertical_attn.train()
    cross_encoder.train()

    early_stopping = EarlyStopping(patience=5, delta=0.0, save_path=f'best_model_div{int(alpha*10)}_sim{int((1-alpha)*10)}.pt')

    for epoch in range(num_epochs):
        total_loss = 0.0
        for table1s, table2s, scores, real_cols1, real_cols2, real_rows1, real_rows2 in train_loader:
            table1s = table1s.to(device)
            table2s = table2s.to(device)
            scores = scores.to(device)
            real_cols1 = real_cols1.to(device)
            real_cols2 = real_cols2.to(device)
            
            
            # 2.1 VerticalSelfAttention -> (B, maxC1, E) / (B, maxC2, E)
            table1_reps = vertical_attn(table1s, real_cols1, real_rows1)  # (B, maxC1, E)
            table2_reps = vertical_attn(table2s, real_cols2, real_rows2)  # (B, maxC2, E)

            # 2.2 CrossEncoder(BERT) -> 로짓, mask까지 고려
            pred_scores = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)

            # 2.3 Loss
            loss = F.mse_loss(pred_scores, scores)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        avg_val_loss = evaluate(vertical_attn, cross_encoder, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.10f}, Val Loss: {avg_val_loss:.10f}")
        
        model_dict = {
            'vertical_attn':vertical_attn.state_dict(),
            'cross_encoder':cross_encoder.state_dict()
        }
        
        early_stopping(avg_val_loss, model_dict)

        if early_stopping.early_stop:
            print('Early stopping is triggered!')
            break

    print("Training finished (or stopped early). Best model loaded.")