import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import RelGraphConv
from sklearn.preprocessing import StandardScaler


# 定义支持多关系类型的RGCN模型
class RGCNWithLearnableAdj(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, num_bases=1, num_nodes=10):
        super(RGCNWithLearnableAdj, self).__init__()

        self.num_rels = num_rels
        self.num_nodes = num_nodes

        # 定义第一个关系图卷积层
        self.conv1 = RelGraphConv(
            in_dim, hidden_dim, num_rels,
            regularizer="basis",
            num_bases=num_bases,
            activation=F.relu
        )

        # 定义第二个关系图卷积层
        self.conv2 = RelGraphConv(
            hidden_dim, out_dim, num_rels,
            regularizer="basis",
            num_bases=num_rels
        )

        # 初始化每种关系类型的邻接矩阵（对称，去除自环）
        adj_initial = torch.ones((num_rels, num_nodes, num_nodes)) - torch.eye(num_nodes).unsqueeze(0).repeat(num_rels,
                                                                                                              1, 1)
        self.adj_matrix = nn.Parameter(adj_initial, requires_grad=True)

    def forward(self, x):
        src_total = []
        dst_total = []
        etype_total = []

        for rel in range(self.num_rels):
            # 对每种关系类型，应用sigmoid以确保边权在0-1之间
            adj = torch.sigmoid(self.adj_matrix[rel])
            s, d = torch.nonzero(adj, as_tuple=True)
            src_total.append(s)
            dst_total.append(d)
            etype_total.append(torch.full_like(s, rel))  # 为每条边分配关系类型

        if len(src_total) > 0:
            src = torch.cat(src_total)
            dst = torch.cat(dst_total)
            etype = torch.cat(etype_total)
            new_edges = (src, dst)

            # 创建新的图
            new_g = dgl.graph(new_edges, num_nodes=self.num_nodes).to(x.device)
            new_g.edata['etype'] = etype
        else:
            # 如果没有边，则创建空图
            new_g = dgl.graph(([], []), num_nodes=self.num_nodes).to(x.device)
            new_g.edata['etype'] = torch.tensor([], dtype=torch.long).to(x.device)

        # 通过第一个关系图卷积层进行前向传播
        h = self.conv1(new_g, x, new_g.edata['etype'])
        # 通过第二个关系图卷积层进行前向传播
        h = self.conv2(new_g, h, new_g.edata['etype'])
        return h


if __name__ == '__main__':
    # 读取数据集
    df = pd.read_csv('dataset/ALFA/dataset1/train.csv')

    # 数据预处理
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop(df[df['label'] == 1].index, axis=0)  # 删除标签为1的故障样本
    df = df.fillna(method='bfill')  # 填充缺失值
    df = df.iloc[0:96, :]  # 取前96行

    # 提取特征名称（从第二列到倒数第二列）
    feature_names = df.columns[1:-1]
    num_features = len(feature_names)  # 特征数量，例如10
    num_timesteps = df.shape[0]  # 时间步数，例如96
    print(f"特征数量: {num_features}, 时间步数: {num_timesteps}")  # 输出特征数量和时间步数
    print(f"特征名称: {feature_names.tolist()}")  # 列出所有特征名称

    # # 构建图结构
    # src, dst = torch.meshgrid(torch.arange(num_features), torch.arange(num_features), indexing='ij')
    # src = src.flatten()
    # dst = dst.flatten()
    # mask = src != dst  # 去除自环
    # src = src[mask]
    # dst = dst[mask]
    # print(f"边的数量: {len(src)}")
    # print(f"源节点列表 (src): {src.tolist()}")
    # print(f"目标节点列表 (dst): {dst.tolist()}")
    #
    # # 定义关系类型（随机分配两种关系类型）
    # # 这里您可以根据实际需求定义关系类型的分配方式
    # rel = np.random.randint(0, 2, size=len(src))  # 随机分配关系类型0或1
    # rel = torch.tensor(rel, dtype=torch.long)
    #
    # # 构建图，并明确指定节点数量为num_features
    # g = dgl.graph((src, dst), num_nodes=num_features)
    # print(f"图中的节点数量: {g.num_nodes()}")  # 应为特征数量，例如10
    #
    # # 赋值关系类型
    # g.edata['etype'] = rel
    # print(f"边的关系类型已赋值: {g.edata['etype']}")

    # 节点特征标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_names].values.seq_len)  # 形状: (特征数量, 时间步数)
    node_features = torch.tensor(scaled_features, dtype=torch.float32)  # 转换为张量
    print(f"节点特征形状 (标准化后): {node_features.shape}")  # 应为 (特征数量, 时间步数)

    # # 赋值节点特征
    # g.ndata['feat'] = node_features
    # print(f"节点特征已成功赋值: {g.ndata['feat'].shape}")  # 应为 (特征数量, 时间步数)

    # 实例化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGCNWithLearnableAdj(
        in_dim=num_timesteps,
        hidden_dim=64,
        out_dim=num_timesteps,
        num_rels=2,  # 设置关系类型数量为2
        num_bases=1,
        num_nodes=num_features
    ).to(device)
    node_features = node_features.to(device)

    # 将数据输入到设备中
    # g = g.to(device)
    # node_features = g.ndata['feat'].to(device)

    # 前向传播
    y = model(x=node_features)
    print(f"输出形状: {y.shape}")  # 应为 (特征数量, 时间步数)
