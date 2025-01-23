import os
import numpy as np
import torch
from queue import PriorityQueue
import pandas as pd
import argparse
from model_distilBERT import VerticalSelfAttention, TableCrossEncoder

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

# Argument parser 설정
parser = argparse.ArgumentParser(description="Run table similarity prediction with DistilBERT.")
parser.add_argument('--k', type=int, default=None, help="Number of top results to display (default: all).")
args = parser.parse_args()

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화
vertical_attn = VerticalSelfAttention(embed_dim=256, expansion_factor=1, num_heads=4, rep_mode="cls")
cross_encoder = TableCrossEncoder(expansion_factor=4, n_layer=6, n_head=8)

# 저장된 가중치 로드
pt_file = 'disbert_lr1e-03_bs32_div3.pt'
checkpoint = torch.load(pt_file, map_location=device)
vertical_attn.load_state_dict(checkpoint['vertical_attn'])
cross_encoder.load_state_dict(checkpoint['cross_encoder'])

# 모델 평가 모드 전환
vertical_attn.eval()
cross_encoder.eval()

# 모델을 device로 이동
vertical_attn.to(device)
cross_encoder.to(device)

# Query 테이블 설정
query_table_path = './input_table/Global Temperature.npy'
query_table = np.load(query_table_path)

# -------- numpy -> tensor --------
query_table = torch.tensor(query_table, dtype=torch.float32).unsqueeze(0)
real_cols1 = torch.tensor([query_table.shape[1]], dtype=torch.int)
real_rows1 = torch.tensor([query_table.shape[2]], dtype=torch.int)

# Target 테이블 리스트 생성
input_table_dir = './input_table'

# CSV 파일 로드
csv_file_path = './balanced_final_df.csv'
df = pd.read_csv(csv_file_path)

# Query 테이블 이름 추출
query_table_name = os.path.basename(query_table_path).replace('.npy', '')

# Query 이름과 일치하는 Target 필터링 & Schema_sim_with_col_datas > 0
filtered_targets = df[(df['Query'] == query_table_name) & (df['Schema_sim_with_col_datas'] > 0)]

# PriorityQueue 생성 (음수 점수를 사용하여 내림차순)
results = PriorityQueue()

# 타겟 테이블들에 대해 예측 실행
for _, row in filtered_targets.iterrows():
    target_file = row['Target'] + '.npy'  # Target 이름으로 파일명 생성
    target_table_path = os.path.join(input_table_dir, target_file)

    # 타겟 파일이 존재하지 않으면 건너뜀
    if not os.path.exists(target_table_path):
        continue

    target_table = np.load(target_table_path)

    # -------- numpy -> tensor --------
    target_table = torch.tensor(target_table, dtype=torch.float32).unsqueeze(0)
    real_cols2 = torch.tensor([target_table.shape[1]], dtype=torch.int)
    real_rows2 = torch.tensor([target_table.shape[2]], dtype=torch.int)

    # IVS 예측
    predict_score = predict(query_table, target_table, real_cols1, real_cols2, real_rows1, real_rows2)

    # PriorityQueue에 음수 점수(-score)와 파일명 저장
    results.put((-predict_score, row['Target']))

# 결과 출력 (내림차순 정렬)
print("\nPrediction scores (descending order):")
count = 0
while not results.empty():
    score, target_name = results.get()
    print(f"File: {target_name}, Score: {-score}")
    count += 1
    if args.k is not None and count >= args.k:  # 상위 k개까지만 출력
        break
