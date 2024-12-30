import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv


class RGCN(nn.Module):
    def __init__(
            self,
            num_nodes,  # 节点的数量
            h_dim,  # 隐藏层的维度
            out_dim,  # 输出层的维度
            num_rels,  # 关系的数量
            regularizer="basis",  # 正则化方法，默认为"basis"
            num_bases=-1,  # 基底的数量，默认为-1，表示使用关系的数量
            dropout=0.0,  # Dropout的比率，默认为0.0
            self_loop=False,  # 是否包含自环，默认为False
            ns_mode=False,  # 是否启用负采样模式，默认为False
    ):
        super(RGCN, self).__init__()  # 调用父类的构造函数

        if num_bases == -1:
            num_bases = num_rels  # 如果基底数量为-1，则将其设置为关系的数量

        self.emb = nn.Embedding(num_nodes, h_dim)  # 节点嵌入层，将节点映射到h_dim维的向量空间
        self.conv1 = RelGraphConv(  # 第一个关系图卷积层
            h_dim,  # 输入特征的维度
            h_dim,  # 输出特征的维度
            num_rels,  # 关系的数量
            regularizer,  # 正则化方法
            num_bases,  # 基底的数量
            self_loop=self_loop  # 是否包含自环
        )
        self.conv2 = RelGraphConv(  # 第二个关系图卷积层
            h_dim,  # 输入特征的维度
            out_dim,  # 输出特征的维度
            num_rels,  # 关系的数量
            regularizer,  # 正则化方法
            num_bases,  # 基底的数量
            self_loop=self_loop  # 是否包含自环
        )
        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        self.ns_mode = ns_mode  # 是否启用负采样模式

    def forward(self, g, nids=None):
        if self.ns_mode:
            # 如果启用负采样模式
            x = self.emb(g[0].srcdata[dgl.NID])  # 获取源节点的嵌入
            h = self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata["norm"])  # 第一个关系图卷积层
            h = self.dropout(F.relu(h))  # 应用ReLU激活函数和Dropout
            h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata["norm"])  # 第二个关系图卷积层
            return h
        else:
            # 如果不启用负采样模式
            x = self.emb.weight if nids is None else self.emb(nids)  # 获取节点嵌入
            h = self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"])  # 第一个关系图卷积层
            h = self.dropout(F.relu(h))  # 应用ReLU激活函数和Dropout
            h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])  # 第二个关系图卷积层
            return h


if __name__ == "__main__":
    import numpy as np

    # 示例参数
    num_nodes = 1000
    original_num_rels = 3  # 原始关系类型数（0, 1, 2）
    h_dim = 128
    out_dim = 64
    dropout = 0.5

    # 创建随机连接的关系图
    num_edges = 5000
    src = np.random.randint(0, num_nodes, size=num_edges)
    dst = np.random.randint(0, num_nodes, size=num_edges)
    etype = np.random.randint(0, original_num_rels, size=num_edges)

    # 创建 DGL 图
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # 添加自环
    g = dgl.add_self_loop(g)  # 总边数为5000 + 1000 = 6000

    # 为自环定义新的关系类型
    self_loop_rel = original_num_rels  # 假设自环为关系类型3
    num_rels = original_num_rels + 1  # 更新关系类型数为4

    # 原始边的 'etype'
    original_etype = torch.tensor(etype, dtype=torch.long)

    # 自环边的 'etype'（新关系类型）
    self_loop_etype = torch.full((num_nodes,), self_loop_rel, dtype=torch.long)

    # 合并 'etype'：首先是原始边的类型，然后是自环边的类型
    combined_etype = torch.cat([original_etype, self_loop_etype])

    # 确保边数匹配
    assert g.number_of_edges() == combined_etype.shape[0], \
        f"边数不匹配：图中有 {g.number_of_edges()} 条边，'etype' 有 {combined_etype.shape[0]} 个元素"

    # 设置图的边特征 'etype'，使用 dgl.ETYPE 常量
    g.edata[dgl.ETYPE] = combined_etype

    # 计算边的规范化因子（基于节点入度）
    deg = g.in_degrees().float().clamp(min=1)
    norm = 1.0 / deg
    g.edata['norm'] = norm[g.edges()[1]].unsqueeze(1)  # 调整形状为 [6000, 1]

    # 初始化 RGCN 模型
    model = RGCN(
        num_nodes=num_nodes,
        h_dim=h_dim,
        out_dim=out_dim,
        num_rels=num_rels,  # 更新后的关系类型数
        regularizer="basis",
        num_bases=-1,  # 默认设置
        dropout=dropout,
        self_loop=False,  # 已经通过 add_self_loop 添加自环
        ns_mode=False
    )

    # 将图和模型移动到相同的设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    g = g.to(device)

    # 前向传播
    output = model(g)  # 输出形状为 [1000, 64]
    print(output.shape)  # 应输出 torch.Size([1000, 64])


# class RGCNWithLearnableAdj(nn.Module):
#     def __init__(self, configs, num_bases=1, num_nodes=10):
#         super(RGCNWithLearnableAdj, self).__init__()
#
#         self.num_rels = configs.num_rels
#         self.num_nodes = num_nodes
#         self.seq_len = configs.seq_len
#         self.hidden_dim = configs.hidden_dim
#
#         # 定义第一个关系图卷积层
#         self.conv1 = RelGraphConv(
#             self.seq_len, self.hidden_dim, self.num_rels,
#             regularizer="basis",
#             num_bases=num_bases,
#             activation=F.relu
#         )
#
#         # 定义第二个关系图卷积层
#         self.conv2 = RelGraphConv(
#             self.hidden_dim, self.seq_len, self.num_rels,
#             regularizer="basis",
#             num_bases=num_rels
#         )
#
#
#         # 初始化每种关系类型的邻接矩阵（对称）
#         adj_initial = torch.ones((self.num_rels, self.num_nodes, self.num_nodes))
#         self.adj_matrix = nn.Parameter(adj_initial, requires_grad=True)
#
#     def forward(self, g, x):
#         """
#         前向传播方法，支持批次输入。
#
#         :param x: 节点特征，形状为 [B, n, in_dim]
#         :return: 输出特征，形状为 [B, n, out_dim]
#         """
#         B, T, d_model = x.shape
#         device = x.device
#
#         # 收集所有批次中的边信息
#         src_total = []
#         dst_total = []
#         etype_total = []
#
#         # 为每种关系类型生成边
#         for rel in range(self.num_rels):
#             # 对每种关系类型，应用sigmoid以确保边权在0-1之间
#             adj = torch.sigmoid(self.adj_matrix[rel]).to(device)  # [n, n]
#             s, d = torch.nonzero(adj, as_tuple=True)
#             src_total.append(s)
#             dst_total.append(d)
#             etype_total.append(torch.full_like(s, rel, dtype=torch.long))
#
#         if len(src_total) > 0:
#             src = torch.cat(src_total)  # [num_edges]
#             dst = torch.cat(dst_total)  # [num_edges]
#             etype = torch.cat(etype_total)  # [num_edges]
#
#             # 为每个批次样本偏移节点索引，避免节点重叠
#             batch_src = []
#             batch_dst = []
#             batch_etype = []
#             for b in range(B):
#                 batch_src.append(src + b * n)
#                 batch_dst.append(dst + b * n)
#                 batch_etype.append(etype)
#
#                 # 合并所有批次的边
#             src_batched = torch.cat(batch_src)  # [B * num_edges]
#             dst_batched = torch.cat(batch_dst)  # [B * num_edges]
#             etype_batched = torch.cat(batch_etype)  # [B * num_edges]
#
#             # 创建一个包含所有批次的图
#             total_nodes = N
#             g = dgl.graph((src_batched, dst_batched), num_nodes=total_nodes).to(device)
#             g.edata['etype'] = etype_batched
#         else:
#             # 如果没有边，则创建空图
#             total_nodes = B * n
#             g = dgl.graph((torch.empty(0, dtype=torch.long).to(device),
#                            torch.empty(0, dtype=torch.long).to(device)), num_nodes=total_nodes).to(device)
#             g.edata['etype'] = torch.empty((0,), dtype=torch.long).to(device)
#
#             # 将输入特征从 [B, n, in_dim] 调整为 [B * n, in_dim]
#         x_flat = x.view(B * n, in_dim)
#
#         # 通过第一个关系图卷积层
#         h = self.conv1(g, x_flat, g.edata['etype'])  # [B * n, hidden_dim]
#         # 通过第二个关系图卷积层
#         h = self.conv2(g, h, g.edata['etype'])  # [B * n, out_dim]
#
#         # 将输出调整回 [B, n, out_dim]
#         h = h.view(B, n, -1)
#
#         return h
#
#     # 示例使用
#
#
# if __name__ == '__main__':
#     # 示例参数
#     B = 4  # 批次大小
#     n = 10  # 节点数
#     in_dim = 96
#     hidden_dim = 64
#     out_dim = 96
#     num_rels = 2
#     num_bases = 1
#
#     # 实例化模型
#     model = RGCNWithLearnableAdj(in_dim, hidden_dim, out_dim, num_rels, num_bases, num_nodes=n)
#
#     # 创建一个批次输入张量 [B, n, in_dim]
#     x = torch.randn(B, n, in_dim)
#
#     # 前向传播
#     output = model(x)  # 输出形状: [B, n, out_dim]
#     print(output.shape)  # 应输出 torch.Size([4, 10, 96])