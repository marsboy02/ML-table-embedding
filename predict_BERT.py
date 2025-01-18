import numpy as np

import torch

from model_BERT import VerticalSelfAttention, TableCrossEncoder

def predict(table1, table2, real_cols1, real_cols2, real_rows1, real_rows2):
    # 데이터를 텐서로 변환하고 device로 이동
    table1 = table1.to(device)
    table2 = table2.to(device)
    real_cols1 = real_cols1.to(device)
    real_cols2 = real_cols2.to(device)
    real_rows1 = real_rows1.to(device)
    real_rows2 = real_rows2.to(device)

    # VerticalSelfAttention을 통해 테이블 임베딩 생성
    with torch.no_grad():
        table1_reps = vertical_attn(table1, real_cols1, real_rows1)  # (1, maxC1, E)
        table2_reps = vertical_attn(table2, real_cols2, real_rows2)  # (1, maxC2, E)

        # CrossEncoder를 통해 예측 점수 생성
        pred_score = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)  # (1,)
    
    return pred_score.item()

# 비교할 테이블 쌍 호출

table1 = np.load('./input_table/Netflix_movies_and_tv_shows_clustering.npy')
table2 = np.load('./input_table/netflix_titles.npy')

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화 (rep_mode = 'cls' or 'mean')

#                               [CLS] vs Mean-Pooling
#                               [CLS] vs Mean-Pooling
#                               [CLS] vs Mean-Pooling

vertical_attn = VerticalSelfAttention(embed_dim=256, expansion_factor=1, num_heads=4, rep_mode="cls")
cross_encoder = TableCrossEncoder(expansion_factor=4, n_layer=6, n_head=8)

#                           =============================
#                           =============================
#                           =============================

# 저장된 가중치 로드

# =========== pt 파일명 지정 ===========
pt_file = ''

checkpoint = torch.load(pt_file, map_location=device, weights_only=True)
vertical_attn.load_state_dict(checkpoint['vertical_attn'])
cross_encoder.load_state_dict(checkpoint['cross_encoder'])

# 모델 평가 모드 전환
vertical_attn.eval()
cross_encoder.eval()

# 모델을 device로 이동
vertical_attn.to(device)
cross_encoder.to(device)

# 테이블 shape 확인
print(table1.shape)
print(table2.shape)

# -------- numpy -> tensor --------
table1 = torch.tensor(table1, dtype = torch.float32).unsqueeze(0)
table2 = torch.tensor(table2, dtype = torch.float32).unsqueeze(0)

real_cols1 = torch.tensor([table1.shape[0]], dtype=torch.int)
real_cols2 = torch.tensor([table2.shape[0]], dtype=torch.int)

real_rows1 = torch.tensor([table1.shape[1]], dtype=torch.int)
real_rows2 = torch.tensor([table2.shape[1]], dtype=torch.int)


# IVS 예측
predict_score = predict(table1, table2, real_cols1, real_cols2, real_rows1, real_rows2)

print(predict_score)