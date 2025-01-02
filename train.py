import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from modeling import MyTableDataset, VerticalSelfAttention, TableCrossEncoder, EarlyStopping
from modeling import collate_fn_with_row_padding, evaluate

print('Dataframe 호출 시작')

df = pd.read_csv('./final_df.csv')
df['IVS'] = 0.91*df['Scaled_Schema_sim'] + 0.09*df['Scaled_Diversity']
tabledata = df[['Query', 'Target', 'IVS']]
tabledata = tabledata.dropna()

print(f'총 학습 데이터 개수 : {tabledata.shape[0]}')

train_df, val_df = train_test_split(tabledata, test_size=0.3, random_state=42)

print('학습 Dataframe 구축 완료')
print(f'train dataset : {train_df.shape[0]}')
print(f'validation dataset : {val_df.shape[0]}')

def train_model(num_epochs, batch_size, lr):
    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device :', device)

    # 하이퍼파라미터
    num_epochs = num_epochs
    batch_size = batch_size
    lr = lr
    
    # 1) 데이터셋 / 데이터로더
    train_dataset = MyTableDataset(train_df, input_table_folder='input_table')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_with_row_padding, shuffle=True)
    
    val_dataset = MyTableDataset(val_df, input_table_folder='input_table')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_with_row_padding, shuffle=True)
    
    # 2) 모델 초기화
    vertical_attn = VerticalSelfAttention(embed_dim=256, num_heads=4, rep_mode="cls")
    cross_encoder = TableCrossEncoder(pretrained_model_name="bert-base-uncased", hidden_dim=256)
    
    # 2-1) 모델 => device
    vertical_attn.to(device)
    cross_encoder.to(device)

    # 옵티마이저
    params = list(vertical_attn.parameters()) + list(cross_encoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    vertical_attn.train()
    cross_encoder.train()
    
    early_stopping = EarlyStopping(patience=5, delta=0.0, save_path='best_model_sche9_div1.pt')

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

train_model(num_epochs=50, batch_size=4, lr=1e-4)