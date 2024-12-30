import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, simpleVIT, Attention_Block, Predict
from layers.KGATBlock import KGATBlock


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs, num_relations):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.num_relations = num_relations  # 新增参数

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                    n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()

        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                KGATBlock(
                    in_channels=configs.c_out,
                    out_channels=configs.d_model,
                    # num_relations=self.num_relations,
                    node_dim=configs.node_dim,
                    dropout=configs.dropout
                )
            )

    def forward(self, x, adj, edge_type):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            # Gconv
            x = self.gconv[i](x, adj[i], edge_type[i])  # [b, t, n] -> [b, t, n]
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

            # for Mul-attetion
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)

            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # Residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs, num_relations):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_relations = num_relations  # 新增参数

        self.model = nn.ModuleList(
            [ScaleGraphBlock(configs, num_relations=self.num_relations) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual, configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)
        self.adj_list = configs.adj_list
        self.edge_type_list = configs.edge_type_list

    def forward(self, x_enc):
        '''
        adj_list: list of adjacency matrices for each ScaleGraphBlock
        edge_type_list: list of relation types for each adjacency matrix
        '''
        # x_enc: [B, T, N]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B, T, C]

        # Pass through ScaleGraphBlocks
        for i in range(self.layer):
            # For each ScaleGraphBlock, pass corresponding adj and edge_type
            adj = self.adj_list[i]  # Assumes adj_list is a list of adjacency matrices
            edge_type = self.edge_type_list[i]  # List of relation types
            enc_out = self.layer_norm(self.model[i](enc_out, adj, edge_type))  # [B, T, C]

        # Project back
        dec_out = self.projection(enc_out)  # [B, T, C] -> [B, T, N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        return dec_out
