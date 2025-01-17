import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertModel
from transformers import DistilBertConfig, DistilBertModel

class MyTableDataset(Dataset):
    """
    주어진 DataFrame(df)에서,
    - 'Query' 열: 예) "employee_table"
    - 'Target' 열: 예) "department_table"
    - 'IVS' 열: float형 스코어

    -> 각각 .npy 파일로 저장된 테이블 임베딩을 로드해 (cols, rows+1, embed_dim) 형태로 반환.
    """
    def __init__(self, df, input_table_folder, transform=None):
        """
        Args:
            df (pandas.DataFrame): 'Query', 'Target', 'IVS' 컬럼을 포함
            input_table_folder (str): .npy 파일이 저장된 폴더 경로
            transform (callable, optional): 불러온 텐서에 적용할 변환 (예: float 형변환, 정규화 등)
        """
        self.df = df.reset_index(drop=True)
        self.input_table_folder = input_table_folder
        self.cache = {}
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def _load_npy(self, table_name):
        if table_name in self.cache:
            return self.cache[table_name]
        else:
            path = os.path.join(self.input_table_folder, table_name + '.npy')
            arr = np.load(path)
            tensor = torch.tensor(arr, dtype = torch.float)
            self.cache[table_name] = tensor
            return tensor

    def __getitem__(self, idx):
        # 1) df에서 'Query', 'Target', 'IVS' 가져오기
        row = self.df.iloc[idx]
        query_name = row['Query']   # 예: "employee_table"
        target_name = row['Target'] # 예: "department_table"
        score = float(row['IVS'])   # float

        table1 = self._load_npy(query_name)
        table2 = self._load_npy(target_name)

        # 4) transform 적용(옵션)
        if self.transform:
            table1 = self.transform(table1)
            table2 = self.transform(table2)

        return table1, table2, score


def collate_fn_with_row_padding(batch):
    """
    (table1, table2, score)을 묶어 batch를 생성.
    - 배치 내 최대 cols, 최대 rows(+1) 모두 찾기
    - 각각 0-padding
    - 최종적으로 (B, max_cols, max_rows, emb_dim) 형태로 stack
    - real_cols 등 필요한 정보를 함께 반환 가능
    """
    table1s, table2s, scores = [], [], []
    real_cols1, real_cols2 = [], []
    real_rows1, real_rows2 = [], []
    # (선택) 만약 실제 row 수도 추적하고 싶다면 real_rows1, real_rows2를 둘 수도 있음.

    # 1) 배치에서 최대 cols/rows 구하기
    max_cols_t1 = 0
    max_rows_t1 = 0
    max_cols_t2 = 0
    max_rows_t2 = 0

    for (t1, t2, s) in batch:
        c1, r1, e1 = t1.shape
        c2, r2, e2 = t2.shape

        # table1
        if c1 > max_cols_t1:
            max_cols_t1 = c1
        if r1 > max_rows_t1:
            max_rows_t1 = r1

        # table2
        if c2 > max_cols_t2:
            max_cols_t2 = c2
        if r2 > max_rows_t2:
            max_rows_t2 = r2

    # 2) 실제 패딩
    for (t1, t2, s) in batch:
        c1, r1, e1 = t1.shape
        c2, r2, e2 = t2.shape

        # 실제 cols(필요하다면 저장)
        real_cols1.append(c1)
        real_cols2.append(c2)

        real_rows1.append(r1)
        real_rows2.append(r2)
        # (만약 실제 rows도 추적하려면 여기서 append)

        # ------------------------
        # (A) table1 row padding
        row_pad_1 = max_rows_t1 - r1
        if row_pad_1 > 0:
            pad_tensor = torch.zeros(c1, row_pad_1, e1)
            # 행(row) 차원은 dim=1
            t1 = torch.cat([t1, pad_tensor], dim=1)  # (c1, max_rows_t1, e1)

        # (B) table1 col padding
        col_pad_1 = max_cols_t1 - c1
        if col_pad_1 > 0:
            pad_tensor = torch.zeros(col_pad_1, max_rows_t1, e1)
            # 컬럼(cols) 차원은 dim=0
            t1 = torch.cat([t1, pad_tensor], dim=0)  # (max_cols_t1, max_rows_t1, e1)

        # ------------------------
        # (C) table2 row padding
        row_pad_2 = max_rows_t2 - r2
        if row_pad_2 > 0:
            pad_tensor = torch.zeros(c2, row_pad_2, e2)
            t2 = torch.cat([t2, pad_tensor], dim=1)  # (c2, max_rows_t2, e2)

        # (D) table2 col padding
        col_pad_2 = max_cols_t2 - c2
        if col_pad_2 > 0:
            pad_tensor = torch.zeros(col_pad_2, max_rows_t2, e2)
            t2 = torch.cat([t2, pad_tensor], dim=0)  # (max_cols_t2, max_rows_t2, e2)

        # 최종적으로 t1: (max_cols_t1, max_rows_t1, e1)
        #            t2: (max_cols_t2, max_rows_t2, e2)

        table1s.append(t1)
        table2s.append(t2)
        scores.append(s)

    # 3) 이제 모든 table1은 (max_cols_t1, max_rows_t1, emb_dim) 동일 shape
    #    table2는 (max_cols_t2, max_rows_t2, emb_dim)
    #    -> stack
    table1s = torch.stack(table1s, dim=0)  # (B, max_cols_t1, max_rows_t1, emb)
    table2s = torch.stack(table2s, dim=0)  # (B, max_cols_t2, max_rows_t2, emb)
    scores = torch.tensor(scores, dtype=torch.float)

    real_cols1 = torch.tensor(real_cols1, dtype=torch.long)  
    real_cols2 = torch.tensor(real_cols2, dtype=torch.long)

    real_rows1 = torch.tensor(real_rows1, dtype=torch.long)
    real_rows2 = torch.tensor(real_rows2, dtype=torch.long)
    

    return table1s, table2s, scores, real_cols1, real_cols2, real_rows1, real_rows2


class VerticalSelfAttention(nn.Module):
    """
    - 입력: (B, max_cols, rows+1, E), 그리고 (B,)짜리 real_cols
    - 컬럼마다 row 시퀀스에 대해 MHA
    - self.rep_mode에 따라 CLS or MEAN
    - 출력: (B, max_cols, E)
    """
    def __init__(self, embed_dim=256, expansion_factor=4, num_heads=4, rep_mode="cls"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rep_mode = rep_mode  # "cls" or "mean"

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.cls_token = torch.zeros(1, 1, embed_dim)

        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.ReLU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x, real_cols, real_rows):
        """
        x: (B, max_cols, R, E)
        real_cols: (B,) 실제 컬럼 개수
        """
        B, maxC, R, E = x.shape
        device = x.device

        rep_list = []

        for b_idx in range(B):
            # 실제 컬럼 수
            ncol = real_cols[b_idx].item()  # 파이썬 int
            nrow = real_rows[b_idx].item()  # 파이썬 int
            
            table_b = x[b_idx]  # (maxC, R, E)

            reps_for_this_table = []
            for col_idx in range(int(ncol)):
                col_tensor = table_b[col_idx]  # (R, E)

                # [CLS] + col_tensor
                cls_token_for_col = self.cls_token.to(device)  # (1,1,E)
                col_tensor = col_tensor.unsqueeze(0)  # (1,R,E)
                col_with_cls = torch.cat([cls_token_for_col, col_tensor], dim=1)  # (1, R+1, E)

                row_mask = torch.zeros(1+R, dtype=torch.bool, device=device)

                row_mask[0] = False
                if nrow > 0:
                    row_mask[1:1+nrow] = False
                if (1 + nrow) < (R + 1):
                    row_mask[1 + nrow:] = True

                row_mask = row_mask.unsqueeze(0)

                # MHA
                attn_out, _ = self.mha(col_with_cls, col_with_cls, col_with_cls, key_padding_mask=row_mask)
                out_ln = self.layernorm(attn_out)
                out_ffn = self.ffn(out_ln)
                out = out_ln + out_ffn

                if self.rep_mode == "cls":
                    rep_vec = out[:, 0, :]  # (1, E)
                else:  # "mean"
                    rep_vec = out[:, 1:1+nrow, :].mean(dim=1)  # (1, E)

                reps_for_this_table.append(rep_vec)  # list of (1, E)

            if ncol > 0:
                reps_for_this_table = torch.cat(reps_for_this_table, dim=0)  # (ncol, E)
            else:
                # 만약 ncol=0이면(이론상), 1xE 0으로
                reps_for_this_table = torch.zeros(1, E).to(device)

            # 나머지 padding 컬럼에 해당하는 부분 (maxC - ncol) 만큼 0으로 채움
            if ncol < maxC:
                pad_cols = maxC - int(ncol)
                pad_tensor = torch.zeros(pad_cols, E).to(device)
                reps_for_this_table = torch.cat([reps_for_this_table, pad_tensor], dim=0)

            # (maxC, E) -> (1, maxC, E)
            rep_list.append(reps_for_this_table.unsqueeze(0))

        reps = torch.cat(rep_list, dim=0)  # (B, maxC, E)
        return reps


class TableCrossEncoder(nn.Module):
    def __init__(self, 
                 expansion_factor=4,
                 n_layer=6,
                 n_head=8,
                 dropout=0.1):
        super().__init__()
        
        config = DistilBertConfig(
            vocab_size=1,
            dim=256,
            hidden_dim=256*expansion_factor,
            n_layers=n_layer,
            n_heads=n_head,
            dropout=dropout
        )
        self.bert = DistilBertModel(config=config)

        # self.projection = nn.Linear(256, 768)

        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1)
        )

    def forward(self, table1_colreps, table2_colreps, real_cols1, real_cols2):
        """
        table1_colreps: (B, maxC1, 256)
        table2_colreps: (B, maxC2, 256)
        real_cols1, real_cols2: (B,) 실제 컬럼 수

        1) [CLS] + table1 + [SEP] + table2
        2) attention_mask: 실제 컬럼 위치는 1, 패딩은 0
        3) BERT forward -> cls_rep -> classifier
        """
        B, maxC1, _ = table1_colreps.shape
        maxC2 = table2_colreps.shape[1]

        # ----- [1] 256 -> 768 변환 -----
        # table1_colreps_768 = self.projection(table1_colreps)  # (B, maxC1, 768)
        # table2_colreps_768 = self.projection(table2_colreps)  # (B, maxC2, 768)


        # ----- [2] [CLS], [SEP] 임베딩 (768차원) -----
        device = table1_colreps.device
        cls_token = torch.zeros(B, 1, 256).to(device)
        sep_token = torch.zeros(B, 1, 256).to(device)

        # sequence: (B, 1 + maxC1 + 1 + maxC2, E)
        sequence = torch.cat([cls_token, table1_colreps, sep_token, table2_colreps], dim=1)

        # attention_mask: same shape (B, seq_len)
        seq_len = sequence.size(1)
        attention_mask = torch.zeros(B, seq_len, dtype=torch.long).to(device)

        # 순서: [CLS] (1개) + table1 (maxC1개) + [SEP] (1개) + table2 (maxC2개)
        # 각 배치별로 실제 col만큼만 mask=1

        for b_idx in range(B):
            # CLS -> always 1
            attention_mask[b_idx, 0] = 1

            # table1 -> real_cols1[b_idx] 개
            r1 = real_cols1[b_idx].item()
            if r1 > 0:
                attention_mask[b_idx, 1 : 1 + int(r1)] = 1

            # SEP -> 1
            sep_pos = 1 + maxC1
            attention_mask[b_idx, sep_pos] = 1

            # table2 -> real_cols2[b_idx] 개
            r2 = real_cols2[b_idx].item()
            if r2 > 0:
                attention_mask[b_idx, sep_pos+1 : sep_pos+1 + int(r2)] = 1

        # ----- [3] BERT forward -----
        outputs = self.bert(
            inputs_embeds=sequence,
            attention_mask=attention_mask
        )
        # cls_rep = outputs.last_hidden_state[:, 0, :]  # (B, 256)

        hidden_states = outputs[0]                      # (B, seq_len, 256)
        cls_rep = hidden_states[:, 0, :]                # (B, 256)

        # ----- [4] Regression -----
        score_pred = self.regressor(cls_rep)  # (B, 1)
        return score_pred.squeeze(-1) # (B, )


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0.0, save_path='best_model.pt'):
        """
        Args:
            patience (int): 성능이 나아지지 않아도 기다릴 epoch 수 (기본값=5)
            delta (float): 개선되었다고 판단할 최소 수치 (기본값=0.0)
            save_path (str): 검증 손실이 갱신될 때 저장할 모델 경로
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_path = save_path

    def __call__(self, val_loss, model_dict):
        """model_dict는 저장할 모델들의 state_dict를 담은 딕셔너리"""
        score = -val_loss  # loss가 낮을수록 성능이 좋은 경우

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_dict):
        """검증 손실이 감소했을 때 모델 가중치를 저장합니다."""
        #print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss
        # model_dict는 {'vertical_attn': vertical_attn.state_dict(),
        #               'cross_encoder': cross_encoder.state_dict()} 등으로 구성
        torch.save(model_dict, self.save_path)


def evaluate(vertical_attn, cross_encoder, val_loader, device):
    vertical_attn.eval()
    cross_encoder.eval()
    
    total_val_loss = 0.0
    with torch.no_grad():
        for table1s, table2s, scores, real_cols1, real_cols2, real_rows1, real_rows2 in val_loader:
            table1s = table1s.to(device)
            table2s = table2s.to(device)
            scores = scores.to(device)
            real_cols1 = real_cols1.to(device)
            real_cols2 = real_cols2.to(device)
            
            # forward
            table1_reps = vertical_attn(table1s, real_cols1, real_rows1)
            table2_reps = vertical_attn(table2s, real_cols2, real_rows2)
            
            pred_scores = cross_encoder(table1_reps, table2_reps, real_cols1, real_cols2)
            
            loss = F.mse_loss(pred_scores, scores)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    # 다시 train 모드로 전환
    vertical_attn.train()
    cross_encoder.train()
    
    return avg_val_loss
