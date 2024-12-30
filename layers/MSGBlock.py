from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.masking import TriangularCausalMask

class Predict(nn.Module):
    def __init__(self,  individual, c_out, seq_len, pred_len, dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len , pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len , pred_len)
            self.dropout = nn.Dropout(dropout)

    #(B,  c_out , seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:,i,:])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out,dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out


class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class GraphBlock(nn.Module):
    def __init__(self, c_out, d_model, conv_channel, skip_channel,
                        gcn_depth, dropout, propalpha, seq_len, node_dim):
        super(GraphBlock, self).__init__()

        # 定义节点向量1
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        # 定义节点向量2
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        # 定义起始卷积层
        self.start_conv = nn.Conv2d(1 , conv_channel, (d_model - c_out + 1, 1))
        # 定义混合传播层
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        # 定义GELU激活函数
        self.gelu = nn.GELU()
        # 定义结束卷积层
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        # 定义线性层
        self.linear = nn.Linear(c_out, d_model)
        # 定义层归一化
        self.norm = nn.LayerNorm(d_model)

    # x in (B, T, d_model)
    # 这里我们使用一个多层感知机来拟合一个复杂的映射f (x)
    def forward(self, x):
        # 计算节点向量1和节点向量2的点积，并进行softmax归一化
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # 对应论文中的Eq.(6), adp in (c_out, c_out)
        # 将输入x进行维度变换
        out = x.unsqueeze(1).transpose(2, 3)  # out in (B, 1, d_model, T)
        # 通过起始卷积层
        out = self.start_conv(out)  # out in (B, conv_channel, N, T)
        # 通过混合传播层
        out = self.gelu(self.gconv1(out, adp))  # out in (B, conv_channel, c_out, T)
        # 通过结束卷积层
        out = self.end_conv(out).squeeze()  # out in (B, T, c_out)
        # 通过线性层
        out = self.linear(out)  # out in (B, T, d_model)

        # 返回归一化后的结果
        return self.norm(x + out)


class nconv(nn.Module):
    def __init__(self):
        # 初始化函数，继承自nn.Module
        super(nconv, self).__init__()

    def forward(self, x, A):
        # forward函数定义了前向传播的计算过程
        # x: 输入张量，形状为 (n, c, w, l)，其中 n 是批量大小，c 是通道数，w 和 l 分别是宽度和长度
        # A: 邻接矩阵，形状为 (v, w)，其中 v 和 w 是节点数

        # 使用torch.einsum进行爱因斯坦求和约定，计算张量乘法
        # 'ncwl,vw->ncvl' 表示：
        # n: 批量大小，保持不变
        # c: 通道数，保持不变
        # w: 输入张量的宽度，与邻接矩阵的行数对应
        # l: 输入张量的长度，保持不变
        # v: 邻接矩阵的列数，表示输出张量的宽度
        x = torch.einsum('ncwl,vw->ncvl', (x, A))

        # 注释掉的代码是另一种可能的计算方式，但未使用
        # 'ncwl,wv->nclv' 表示：
        # n: 批量大小，保持不变
        # c: 通道数，保持不变
        # w: 输入张量的宽度，与邻接矩阵的列数对应
        # l: 输入张量的长度，保持不变
        # v: 邻接矩阵的行数，表示输出张量的长度
        # x = torch.einsum('ncwl,wv->nclv', (x, A))

        # 返回连续的张量，确保内存布局是连续的，这对后续操作可能有优化作用
        return x.contiguous()


class linear(nn.Module):
    # 定义一个名为linear的类，继承自nn.Module，用于实现一个线性层

    def __init__(self, c_in, c_out, bias=True):
        # 构造函数，初始化线性层
        super(linear, self).__init__()
        # 调用父类nn.Module的构造函数，确保正确初始化

        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)
        # 创建一个二维卷积层（实际上是一个1x1卷积），用于实现线性变换
        # 参数c_in表示输入通道数，c_out表示输出通道数
        # kernel_size=(1, 1)表示卷积核大小为1x1，即进行线性变换
        # padding=(0,0)表示不进行填充
        # stride=(1,1)表示步长为1
        # bias=True表示是否添加偏置项

    def forward(self, x):
        # 定义前向传播函数
        return self.mlp(x)
        # 将输入x通过mlp（即1x1卷积层）进行线性变换，并返回结果


class mixprop(nn.Module):
    # 定义一个混合传播层类
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        # 初始化函数，传入输入特征数、输出特征数、传播深度、dropout率和alpha值
        super(mixprop, self).__init__()
        # 调用父类初始化函数
        self.nconv = nconv()
        # 定义一个卷积层
        self.mlp = linear((gdep+1)*c_in, c_out)
        # 定义一个全连接层
        self.gdep = gdep
        # 保存传播深度
        self.dropout = dropout
        # 保存dropout率
        self.alpha = alpha
        # 保存alpha值

    def forward(self, x, adj):
        # 定义前向传播函数，传入输入特征和邻接矩阵
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        # 在邻接矩阵中添加自环
        d = adj.sum(1)
        # 计算度矩阵
        h = x
        # 初始化特征
        out = [h]
        # 将初始特征加入输出列表
        a = adj / d.view(-1, 1)
        # 计算归一化的邻接矩阵
        for i in range(self.gdep):
            # 循环传播深度次
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
            # 更新特征
            out.append(h)
        # 将所有特征拼接
        ho = torch.cat(out,dim=1)
        # 将特征输入全连接层
        ho = self.mlp(ho)
        # 返回输出特征
        return ho


class simpleVIT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size=2, depth=1, num_heads=4, dropout=0.1,init_weight =True):
        super(simpleVIT, self).__init__()
        self.emb_size = emb_size
        self.depth = depth
        self.to_patch = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 2 * patch_size + 1, padding= patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                FeedForward(emb_size,  emb_size)
            ]))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        B , N ,_ ,P = x.shape
        x = self.to_patch(x)
        # x = x.permute(0, 2, 3, 1).reshape(B,-1, N)
        for  norm ,attn, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x

        x = x.transpose(1,2).reshape(B, self.emb_size ,-1, P)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)