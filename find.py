import os
import numpy as np
import torch
from queue import PriorityQueue
import pandas as pd
import argparse
from model_BERT import VerticalSelfAttention, TableCrossEncoder
# from model_distilBERT import VerticalSelfAttention, TableCrossEncoder

# ===============================================================
# 1) 조인 가능 컬럼 탐지 함수
def detect_joinable_columns(df1, df2):
    """
    df1, df2: pandas DataFrame
    숫자형 컬럼은 제외하고, 문자열 형태로 변환했을 때 공통값이 존재하면 조인 가능 컬럼(pair)로 간주
    """
    joinable_pairs = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(df1[col1]) or pd.api.types.is_numeric_dtype(df2[col2]):
                continue
            
            # Check if there is any overlap in string-converted values
            if df1[col1].astype(str).isin(df2[col2].astype(str)).any():
                joinable_pairs.append((col1, col2))
    
    return joinable_pairs

# ===============================================================
# 2) 저장된 가중치 로드
pt_file = 'models/bert_cls_base_lr1e-03_bs32_div7.pt'

# Query 테이블 설정 (.npy 파일)
query_table_path = './input_table/Countries Capitals and their coordinates - Sheet1.npy'
query_table = np.load(query_table_path)

# Target 테이블 리스트 생성
input_table_dir = './input_table'

# CSV 파일 로드 (IVS 등 스코어가 계산된 중간 결과)
csv_file_path = './balanced_final_df.csv'
df = pd.read_csv(csv_file_path)

# ===============================================================
# 3) IVS 계산 함수

def calculate_ivs_and_sort(df, pt_file):
    # 예시: pt_file[-4] 위치한 숫자를 factor로 사용 (가중치 가정)
    pt_file_factor = int(pt_file[-4])

    # Calculate IVS
    df['IVS'] = (
        df['scaled_diversity'] * 0.1 * pt_file_factor +
        df['scaled_cosine_sim'] * 0.1 * (10 - pt_file_factor)
    )

    # Sort by IVS in descending order
    return df.sort_values(by='IVS', ascending=False)

# ===============================================================
# 4) Prediction 함수 (예시)

def predict(table1, table2, real_cols1, real_cols2, real_rows1, real_rows2):
    table1 = table1.to(device)
    table2 = table2.to(device)
    real_cols1 = real_cols1.to(device)
    real_cols2 = real_cols2.to(device)
    real_rows1 = real_rows1.to(device)
    real_rows2 = real_rows2.to(device)

    with torch.no_grad():
        table1_reps = vertical_attn(table1, real_cols1, real_rows1)
        table2_reps = vertical_attn(table2, real_cols2, real_rows2)

        pred_score = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)

    return pred_score.item()

# ===============================================================
# 5) Argument parser 설정
parser = argparse.ArgumentParser(description="Run table similarity prediction.")
parser.add_argument('--k', type=int, default=None, help="Number of top results to display (default: all).")
args = parser.parse_args()

# GPU/CPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화
vertical_attn = VerticalSelfAttention(embed_dim=256, expansion_factor=4, num_heads=4, rep_mode="cls")
cross_encoder = TableCrossEncoder(expansion_factor=4, n_layer=6, n_head=8)

checkpoint = torch.load(pt_file, map_location=device)
vertical_attn.load_state_dict(checkpoint['vertical_attn'])
cross_encoder.load_state_dict(checkpoint['cross_encoder'])

vertical_attn.eval()
cross_encoder.eval()
vertical_attn.to(device)
cross_encoder.to(device)

# -------- numpy -> tensor --------
query_table = torch.tensor(query_table, dtype=torch.float32).unsqueeze(0)
real_cols1 = torch.tensor([query_table.shape[1]], dtype=torch.int)
real_rows1 = torch.tensor([query_table.shape[2]], dtype=torch.int)

# Query 테이블 이름 추출 (예: Country_data.npy -> Country_data)
query_table_name = os.path.basename(query_table_path).replace('.npy', '')

# Query 이름과 일치하는 Target 필터링 & Schema_sim_with_col_datas > 0
filtered_targets = df[(df['Query'] == query_table_name) & (df['Schema_sim_with_col_datas'] > 0)]

# PriorityQueue 생성 (모델 기반 점수)
results = PriorityQueue()

# ===============================================================
# 6) 타겟 테이블들에 대해 모델 예측 점수 계산
for _, row in filtered_targets.iterrows():
    target_file = row['Target'] + '.npy'
    target_table_path = os.path.join(input_table_dir, target_file)

    if not os.path.exists(target_table_path):
        continue

    target_table = np.load(target_table_path)
    target_table = torch.tensor(target_table, dtype=torch.float32).unsqueeze(0)
    real_cols2 = torch.tensor([target_table.shape[1]], dtype=torch.int)
    real_rows2 = torch.tensor([target_table.shape[2]], dtype=torch.int)

    predict_score = predict(query_table, target_table, real_cols1, real_cols2, real_rows1, real_rows2)

    results.put((-predict_score, row['Target']))

# ===============================================================
# 7) IVS 계산 및 정렬
sorted_targets = calculate_ivs_and_sort(filtered_targets, pt_file)

# ===============================================================
# 8) 출력 단계: IVS 상위 K개 + 모델 예측 점수 상위 K개
#    --> 여기서 CSV를 추가로 로드하여, joinable columns를 검사하는 로직을 추가

print("\n[IVS-based ranking (descending order)]")
for i, (_, row) in enumerate(sorted_targets.iterrows(), 1):
    target_name = row['Target']
    ivs_score = row['IVS']

    print(f"{i}. File: {target_name}, IVS Score: {ivs_score:.6f}")

    # -------------------------------
    # (1) Query CSV와 Target CSV를 로드하여 joinable columns 확인
    query_csv_path = os.path.join('dataset', query_table_name + '.csv')   # 사용자 환경에 맞게 수정
    target_csv_path = os.path.join('dataset', target_name + '.csv')      # 사용자 환경에 맞게 수정

    if os.path.exists(query_csv_path) and os.path.exists(target_csv_path):
        df_query_csv = pd.read_csv(query_csv_path)
        df_target_csv = pd.read_csv(target_csv_path)

        joinable_cols = detect_joinable_columns(df_query_csv, df_target_csv)
        if joinable_cols:
            print(f"   -> Joinable columns: {joinable_cols}")
        else:
            print(f"   -> No string-based overlapping columns found.")
    else:
        print("   -> CSV files not found, skipping joinable columns check.")

    if args.k is not None and i >= args.k:
        break

print("\n[Prediction scores (descending order)]")
count = 0
while not results.empty():
    score, target_name = results.get()
    # score는 PriorityQueue에서 넣을 때 음수화(-predict_score)했으므로 다시 -score
    model_score = -score
    count += 1

    print(f"{count}. File: {target_name}, Model Score: {model_score:.6f}", end='')

    # -------------------------------
    # (2) Query CSV와 Target CSV를 로드하여 joinable columns 확인
    query_csv_path = os.path.join('dataset', query_table_name + '.csv')   # 사용자 환경에 맞게 수정
    target_csv_path = os.path.join('dataset', target_name + '.csv')      # 사용자 환경에 맞게 수정

    if os.path.exists(query_csv_path) and os.path.exists(target_csv_path):
        df_query_csv = pd.read_csv(query_csv_path)
        df_target_csv = pd.read_csv(target_csv_path)

        joinable_cols = detect_joinable_columns(df_query_csv, df_target_csv)
        if joinable_cols:
            print(f"   -> Joinable columns: {joinable_cols}")
        else:
            print(f"   -> No string-based overlapping columns found.")
    else:
        print("   -> CSV files not found, skipping joinable columns check.")

    if args.k is not None and count >= args.k:
        break
