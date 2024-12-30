import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import RelGraphConv
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
# 请确保 'dataset/ALFA/dataset1/train.csv' 路径下存在您的数据集
df = pd.read_csv('dataset/ALFA/dataset1/train.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 删除标签为1的故障样本
df = df.drop(df[df['label'] == 1].index, axis=0)

# 填充缺失值
df = df.fillna(method='bfill')

# 取前20行
df = df.iloc[0:20, :]

# 提取特征名称（从第二列到倒数第二列）
feature_names = df.columns[1:-1]
num_features = len(feature_names)  # 特征数量，例如10
num_timesteps = df.shape[0]       # 时间步数，例如20

print(f"特征数量: {num_features}, 时间步数: {num_timesteps}")  # 输出特征数量和时间步数
print(f"特征名称: {feature_names.tolist()}")  # 列出所有特征名称

# 2. 构建初始图结构（可以初始化为全连接图，后续通过邻接矩阵学习调整）
# 使用 torch.meshgrid 并指定 indexing='ij' 以避免未来版本的兼容性问题
src, dst = torch.meshgrid(torch.arange(num_features), torch.arange(num_features), indexing='ij')
src = src.flatten()
dst = dst.flatten()
mask = src != dst  # 去除自环
src = src[mask]
dst = dst[mask]

print(f"边的数量: {len(src)}")
print(f"源节点列表 (src): {src.tolist()}")
print(f"目标节点列表 (dst): {dst.tolist()}")

# 定义关系类型（单一关系类型）
rel = np.zeros(len(src), dtype=np.int64)  # 关系类型为0

# 构建图，并明确指定节点数量为num_features
g = dgl.graph((src, dst), num_nodes=num_features)
print(f"图中的节点数量: {g.num_nodes()}")  # 应为特征数量，例如10

g.edata['etype'] = torch.tensor(rel, dtype=torch.long)

# 3. 构造节点特征
# 特征标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_names].values.seq_len)  # 形状: (特征数量, 时间步数)

node_features = torch.tensor(scaled_features, dtype=torch.float32)  # 转换为张量
print(f"节点特征形状 (标准化后): {node_features.shape}")  # 应为 (特征数量, 时间步数)

# 赋值节点特征
g.ndata['feat'] = node_features
print(f"节点特征已成功赋值: {g.ndata['feat'].shape}")  # 应为 (特征数量, 时间步数)

# 4. 定义带可学习邻接矩阵的RGCN模型
class RGCNWithLearnableAdj(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, num_bases=1, num_nodes=10):
        super(RGCNWithLearnableAdj, self).__init__()
        self.conv1 = RelGraphConv(
            in_dim, hidden_dim, num_rels,
            regularizer="basis",
            num_bases=num_bases,
            activation=F.relu
        )
        self.conv2 = RelGraphConv(
            hidden_dim, out_dim, num_rels,
            regularizer="basis",
            num_bases=num_rels
        )
        # 初始化可学习的邻接矩阵（对称，去除自环）
        adj_initial = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
        self.adj_matrix = nn.Parameter(adj_initial, requires_grad=True)

    def forward(self, g, x, edge_type):
        # 使用当前的邻接矩阵构建新的边列表
        adj = torch.sigmoid(self.adj_matrix)  # 确保边权在0-1之间
        src, dst = torch.nonzero(adj, as_tuple=True)
        new_edges = (src, dst)

        # 创建新的图
        new_g = dgl.graph(new_edges, num_nodes=g.num_nodes()).to(x.device)
        # 由于num_rels=1，所有边类型都为0
        new_g.edata['etype'] = torch.zeros(len(src), dtype=torch.long).to(x.device)

        h = self.conv1(new_g, x, new_g.edata['etype'])
        h = self.conv2(new_g, h, new_g.edata['etype'])
        return h

# 5. 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCNWithLearnableAdj(in_dim=num_timesteps, hidden_dim=64, out_dim=num_timesteps,
                            num_rels=1, num_bases=1, num_nodes=num_features).to(device)
g = g.to(device)
node_features = g.ndata['feat'].to(device)

# 6. 无监督训练示例（节点特征重构）
class RGCNAutoEncoderWithLearnableAdj(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, num_bases=1, num_nodes=10):
        super(RGCNAutoEncoderWithLearnableAdj, self).__init__()
        self.encoder = RGCNWithLearnableAdj(in_dim, hidden_dim, hidden_dim, num_rels, num_bases, num_nodes)
        self.decoder = RGCNWithLearnableAdj(hidden_dim, hidden_dim, out_dim, num_rels, num_bases, num_nodes)

    def forward(self, g, x, edge_type):
        encoded = self.encoder(g, x, edge_type)
        decoded = self.decoder(g, encoded, edge_type)
        return decoded

autoencoder = RGCNAutoEncoderWithLearnableAdj(in_dim=num_timesteps, hidden_dim=64, out_dim=num_timesteps,
                                             num_rels=1, num_bases=1, num_nodes=num_features).to(device)

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

num_epochs = 100
lambda_reg = 1e-4  # 正则化系数，防止邻接矩阵过度调整

for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()

    reconstructed = autoencoder(g, node_features, g.edata['etype'])

    loss = loss_fn(reconstructed, node_features)
    # 添加邻接矩阵的正则化项（L1正则化）
    l1_reg = torch.norm(autoencoder.encoder.adj_matrix, 1) + torch.norm(autoencoder.decoder.adj_matrix, 1)
    total_loss = loss + lambda_reg * l1_reg
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, L1 Reg: {l1_reg.item():.4f}")

# 获取编码后的节点表示
autoencoder.eval()
with torch.no_grad():
    encoded_features = autoencoder.encoder(g, node_features, g.edata['etype'])
    print(f"编码后的节点特征形状: {encoded_features.shape}")  # 应为 (特征数量, 64)

# 7. 有监督训练示例（分类任务）
# 注意：当前代码生成了随机标签，导致验证和测试准确率为0。请使用真实标签进行训练。
# 生成随机标签作为示例（请根据实际任务替换）
num_classes = 3
labels = torch.randint(0, num_classes, (num_features,)).to(device)
g.ndata['label'] = labels

class RGCNClassifierWithLearnableAdj(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_rels, num_bases=1, num_nodes=10):
        super(RGCNClassifierWithLearnableAdj, self).__init__()
        self.conv1 = RelGraphConv(
            in_dim, hidden_dim, num_rels,
            regularizer="basis",
            num_bases=num_bases,
            activation=F.relu
        )
        self.conv2 = RelGraphConv(
            hidden_dim, num_classes, num_rels,
            regularizer="basis",
            num_bases=num_rels
        )
        # 初始化可学习的邻接矩阵（对称，去除自环）
        adj_initial = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
        self.adj_matrix = nn.Parameter(adj_initial, requires_grad=True)

    def forward(self, g, x, edge_type):
        # 使用当前的邻接矩阵构建新的边列表
        adj = torch.sigmoid(self.adj_matrix)  # 确保边权在0-1之间
        src, dst = torch.nonzero(adj, as_tuple=True)
        new_edges = (src, dst)

        # 创建新的图
        new_g = dgl.graph(new_edges, num_nodes=g.num_nodes()).to(x.device)
        # 由于num_rels=1，所有边类型都为0
        new_g.edata['etype'] = torch.zeros(len(src), dtype=torch.long).to(x.device)

        h = self.conv1(new_g, x, new_g.edata['etype'])
        h = self.conv2(new_g, h, new_g.edata['etype'])
        return h

# 初始化分类器模型
classifier = RGCNClassifierWithLearnableAdj(in_dim=num_timesteps, hidden_dim=64, num_classes=num_classes,
                                           num_rels=1, num_bases=1, num_nodes=num_features).to(device)

optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 数据划分
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 计算各集数量
train_size = int(train_ratio * num_features)
val_size = int(val_ratio * num_features)
test_size = num_features - train_size - val_size

indices = torch.randperm(num_features)
train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

print(f"训练集索引: {train_idx.tolist()}")
print(f"验证集索引: {val_idx.tolist()}")
print(f"测试集索引: {test_idx.tolist()}")

num_epochs = 100
lambda_reg = 1e-4  # 正则化系数，防止邻接矩阵过度调整

for epoch in range(num_epochs):
    classifier.train()
    optimizer.zero_grad()

    logits = classifier(g, node_features, g.edata['etype'])

    loss = loss_fn(logits[train_idx], labels[train_idx])
    # 添加邻接矩阵的正则化项（L1正则化）
    l1_reg = torch.norm(classifier.adj_matrix, 1)
    total_loss = loss + lambda_reg * l1_reg
    total_loss.backward()
    optimizer.step()

    # 验证集评估
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(g, node_features, g.edata['etype'])
        val_preds = val_logits[val_idx].argmax(dim=1)
        val_labels = labels[val_idx]
        val_acc = (val_preds == val_labels).float().mean().item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, L1 Reg: {l1_reg.item():.4f}")

# 测试模型
classifier.eval()
with torch.no_grad():
    test_logits = classifier(g, node_features, g.edata['etype'])
    test_preds = test_logits[test_idx].argmax(dim=1)
    test_labels = labels[test_idx]
    test_acc = (test_preds == test_labels).float().mean().item()
    print(f"测试集准确率: {test_acc:.4f}")

# 获取分类模型的中间层表示
class RGCNFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(RGCNFeatureExtractor, self).__init__()
        self.model = model

    def forward(self, g, x, edge_type):
        adj = torch.sigmoid(self.model.adj_matrix)
        src, dst = torch.nonzero(adj, as_tuple=True)
        new_edges = (src, dst)
        new_g = dgl.graph(new_edges, num_nodes=g.num_nodes()).to(x.device)
        # 由于num_rels=1，所有边类型都为0
        new_g.edata['etype'] = torch.zeros(len(src), dtype=torch.long).to(x.device)
        h = self.model.conv1(new_g, x, new_g.edata['etype'])
        h = self.model.conv2(new_g, h, new_g.edata['etype'])
        return h

feature_extractor = RGCNFeatureExtractor(classifier).to(device)
feature_extractor.eval()

with torch.no_grad():
    feature_representation = feature_extractor(g, node_features, g.edata['etype'])
    print(f"特征表示形状: {feature_representation.shape}")  # 应为 (特征数量, 64)