import torch
import torch.nn as nn

class VerticalSelfAttention(nn.Module):
    def __init__(self, embed_dim = 256, num_heads = 4, rep_mode = 'mean'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rep_mode = rep_mode

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, real_cols, real_rows):
        B, maxC, R, E = x.shape
        device = x.device

        rep_list = []

        for b_idx in range(B):
            ncol = real_cols[b_idx].item()
            nrow = real_rows[b_idx].item()

            table_b = x[b_idx]

            reps_for_this_table = []
            
            for col_idx in range(int(ncol)):
                col_tensor = table_b[col_idx]
                col_tensor = col_tensor.unsqueeze(0)

                row_mask = torch.zeros(R, dtype=torch.bool, device=device)

                if nrow < R:
                    row_mask[nrow:] = True
                row_mask = row_mask.unsqueeze(0)

                attn_out, _ = self.mha(
                    col_tensor,
                    col_tensor,
                    col_tensor,
                    key_padding_mask=row_mask
                )

                out_ln = self.layernorm(attn_out)
                out_ffn = self.ffn(out_ln)
                out = out_ln + out_ffn

                if self.rep_mode == 'mean':
                    rep_vec = out[:, :nrow, :].mean(dim=1)

                reps_for_this_table.append(rep_vec)

            if ncol > 0:
                reps_for_this_table = torch.cat(reps_for_this_table, dim = 0)
            else:
                reps_for_this_table = torch.zeros(1, E).to(device)

            if ncol < maxC:
                pad_cols = maxC - int(ncol)
                pad_tensor = torch.zeros(pad_cols, E).to(device)
                reps_for_this_table = torch.cat([reps_for_this_table, pad_tensor], dim=0)
            
            rep_list.append(reps_for_this_table.unsqueeze(0))

        reps = torch.cat(rep_list, dim=0)
        return reps