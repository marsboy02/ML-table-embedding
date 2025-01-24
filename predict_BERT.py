import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ========= (1) CSV에서 데이터 불러오기 & IVS 계산/Scaling ==========

df = pd.read_csv('./balanced_final_df.csv')
df = df.dropna()

# 저장된 가중치 로드 (.pt 파일)
pt_file = 'bert_lr1e-03_bs64_div7.pt'

alpha = 0.7  # 예: 0.1, 0.3, 0.5, 0.7, 0.9 등
df['ivs'] = alpha * df['scaled_diversity'] + (1 - alpha) * df['scaled_cosine_sim']

scaler = MinMaxScaler()
df['IVS'] = scaler.fit_transform(df[['ivs']])  # IVS 컬럼 생성

# 예시로 Query, Target, IVS만 사용한다고 가정
tabledata = df[['Query', 'Target', 'IVS']]
tabledata = tabledata.dropna()

# ========= (2) train/val split =========
train_df, val_df = train_test_split(tabledata, test_size=0.3, random_state=42)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))

# ========= (3) 모델 불러오기 및 predict 함수 선언 ==========

# -- 모델 구조 임포트 (model_BERT.py 등)
from model_BERT import VerticalSelfAttention, TableCrossEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화 (rep_mode='cls' 예시)
vertical_attn = VerticalSelfAttention(embed_dim=256, expansion_factor=4, num_heads=4, rep_mode="cls")
cross_encoder = TableCrossEncoder(expansion_factor=4, n_layer=6, n_head=8)

checkpoint = torch.load(pt_file, map_location=device)
vertical_attn.load_state_dict(checkpoint['vertical_attn'])
cross_encoder.load_state_dict(checkpoint['cross_encoder'])

vertical_attn.eval().to(device)
cross_encoder.eval().to(device)

# -- predict 함수 (질문에 주어진 코드와 동일/유사)
def predict(table1, table2, real_cols1, real_cols2, real_rows1, real_rows2):
    table1 = table1.to(device)
    table2 = table2.to(device)
    real_cols1 = real_cols1.to(device)
    real_cols2 = real_cols2.to(device)
    real_rows1 = real_rows1.to(device)
    real_rows2 = real_rows2.to(device)

    with torch.no_grad():
        # VerticalSelfAttention 적용
        table1_reps = vertical_attn(table1, real_cols1, real_rows1)  # (1, maxC1, E)
        table2_reps = vertical_attn(table2, real_cols2, real_rows2)  # (1, maxC2, E)
        # CrossEncoder로 최종 점수 예측
        pred_score = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)  # (1,)

    return pred_score.item()

# ========= (4) Validation 세트에 대해 테이블 로드 후 예측 ==========

def predict_row(row):
    """
    val_df의 한 행(row)에 대해:
      - row['Query'], row['Target'] 로부터 .npy 파일 로드
      - predict 함수를 호출하여 결과(예측 점수) 반환
    """
    query_name = row['Query']  # 예: 'Netflix_movies_and_tv_shows_clustering'
    target_name = row['Target']  # 예: 'netflix_titles'
    
    # 실제 .npy 파일명 규칙에 맞게 경로 수정
    table1_path = f'./input_table/{query_name}.npy'
    table2_path = f'./input_table/{target_name}.npy'
    
    # npy 파일 로드
    table1 = np.load(table1_path)
    table2 = np.load(table2_path)
    
    # 텐서 변환 (batch 차원 추가)
    table1_tensor = torch.tensor(table1, dtype=torch.float32).unsqueeze(0)
    table2_tensor = torch.tensor(table2, dtype=torch.float32).unsqueeze(0)
    
    real_cols1 = torch.tensor([table1_tensor.shape[0]], dtype=torch.int)
    real_cols2 = torch.tensor([table2_tensor.shape[0]], dtype=torch.int)
    real_rows1 = torch.tensor([table1_tensor.shape[1]], dtype=torch.int)
    real_rows2 = torch.tensor([table2_tensor.shape[1]], dtype=torch.int)
    
    # 예측 점수
    pred_score = predict(table1_tensor, 
                         table2_tensor, 
                         real_cols1, 
                         real_cols2, 
                         real_rows1, 
                         real_rows2)
    return pred_score

# Validation DataFrame에 'pred_IVS' 컬럼 추가
val_df = val_df.copy()
val_df['pred_IVS'] = val_df.apply(predict_row, axis=1)

# ========= (5) Validation 결과 확인 =========
print(val_df.head())
top_15_by_pred = val_df.sort_values(by='pred_IVS', ascending=False).head(15)
print(top_15_by_pred)


# 실제 IVS와의 비교(상관계수, RMSE 등)도 가능
correlation = val_df[['IVS','pred_IVS']].corr().iloc[0,1]
print("Correlation between true IVS and predicted IVS:", correlation)
