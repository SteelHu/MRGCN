import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregator(nn.Module):
    """
    特征聚合类，支持多种聚合方法。

    支持的聚合方法：
        - 'sum': 元素级求和
        - 'average': 元素级平均
        - 'max': 元素级最大值
        - 'weighted_sum': 可学习的加权求和
        - 'attention': 注意力机制
        - 'mlp': 多层感知机
        - 'conv_fusion': 卷积融合

    参数:
        method (str): 聚合方法名称。
        num_rels (int): 关系类型的数量。
        embed_size (int, optional): 注意力机制或其他需要的嵌入尺寸。
        hidden_dim (int, optional): MLP 的隐藏层尺寸。
        output_dim (int, optional): MLP 的输出尺寸。
        kernel_size (int, optional): 卷积核大小（仅适用于卷积融合）。
    """

    def __init__(self, method, num_rels, embed_size=None, hidden_dim=None, output_dim=None, kernel_size=3):
        super(FeatureAggregator, self).__init__()
        self.method = method.lower()
        self.num_rels = num_rels

        if self.method == 'weighted_sum':
            # 初始化可学习的权重参数
            self.weights = nn.Parameter(torch.ones(num_rels))

        elif self.method == 'attention':
            assert embed_size is not None, "embed_size 必须为注意力机制提供。"
            self.attention = nn.Sequential(
                nn.Linear(embed_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        elif self.method == 'mlp':
            assert hidden_dim is not None and output_dim is not None, "hidden_dim 和 output_dim 对于 MLP 是必须的。"
            # 假设输入特征维度为 num_rels * feature_dim
            self.mlp = nn.Sequential(
                nn.Linear(num_rels * 128, hidden_dim),  # 假设每个特征维度为128
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        elif self.method == 'conv_fusion':
            assert output_dim is not None, "output_dim 对于卷积融合是必须的。"
            self.conv = nn.Conv1d(num_rels, output_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        elif self.method not in ['sum', 'average', 'max']:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

    def forward(self, ho):
        """
        前向传播。

        参数:
            ho (list of torch.Tensor): 包含不同关系类型的特征张量列表。
                                      每个张量形状为 (B, C, N, L)。

        返回:
            torch.Tensor: 聚合后的特征张量，形状根据聚合方法不同而不同。
        """
        if self.method in ['sum', 'average', 'max']:
            stacked_ho = torch.stack(ho, dim=0)  # 形状: (num_rels, B, C, N, L)
            if self.method == 'sum':
                final_features = stacked_ho.sum(dim=0)  # 形状: (B, C, N, L)
            elif self.method == 'average':
                final_features = stacked_ho.mean(dim=0)  # 形状: (B, C, N, L)
            elif self.method == 'max':
                final_features, _ = stacked_ho.max(dim=0)  # 形状: (B, C, N, L)

        elif self.method == 'weighted_sum':
            # 可学习的加权求和
            stacked_ho = torch.stack(ho, dim=0)  # 形状: (num_rels, B, C, N, L)
            weights = F.softmax(self.weights, dim=0).view(self.num_rels, 1, 1, 1, 1)  # 形状: (num_rels, 1, 1, 1, 1)
            final_features = (stacked_ho * weights).sum(dim=0)  # 形状: (B, C, N, L)

        elif self.method == 'attention':
            # 注意力机制聚合
            B, C, N, L = ho[0].shape
            stacked_ho = torch.stack(ho, dim=1)  # 形状: (B, num_rels, C, N, L)

            # Permute to (B, N, L, num_rels, C)
            stacked_ho = stacked_ho.permute(0, 3, 4, 1, 2).contiguous()  # (B, N, L, num_rels, C)
            # Reshape to (B * N * L, num_rels, C)
            ho_reshape = stacked_ho.view(B * N * L, self.num_rels, C)  # (B*N*L, num_rels, C)

            # 计算注意力分数
            attn_scores = self.attention(ho_reshape)  # 形状: (B*N*L, num_rels, 1)
            attn_scores = attn_scores.squeeze(-1)  # (B*N*L, num_rels)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B*N*L, num_rels)

            # 应用注意力权重
            attn_weights = attn_weights.unsqueeze(-1)  # (B*N*L, num_rels, 1)
            weighted_features = ho_reshape * attn_weights  # (B*N*L, num_rels, C)
            final_features = weighted_features.sum(dim=1)  # (B*N*L, C)

            # Reshape back to (B, C, N, L)
            final_features = final_features.view(B, N, L, C).permute(0, 3, 1, 2).contiguous()  # (B, C, N, L)

        elif self.method == 'mlp':
            # MLP 聚合
            B, C, N, L = ho[0].shape
            stacked_ho = torch.stack(ho, dim=1)  # (B, num_rels, C, N, L)
            # Reshape to (B, num_rels, C * N * L)
            ho_reshape = stacked_ho.view(B, self.num_rels, -1)  # (B, num_rels, C*N*L)
            # Flatten num_rels and C*N*L for MLP input
            ho_reshape = ho_reshape.view(B, self.num_rels * C * N * L)  # (B, num_rels * C * N * L)
            # 通过 MLP
            final_features = self.mlp(ho_reshape)  # (B, output_dim)
            # 假设希望最终特征为 (B, output_dim, N, L)，需要进一步处理
            final_features = final_features.view(B, -1, 1, 1)  # (B, output_dim, 1, 1)
            # 扩展到 (B, output_dim, N, L)
            final_features = final_features.expand(-1, -1, N, L)  # (B, output_dim, N, L)

        elif self.method == 'conv_fusion':
            # 卷积融合聚合
            B, C, N, L = ho[0].shape
            stacked_ho = torch.stack(ho, dim=2)  # (B, C, num_rels, N, L)
            # Reshape to (B * C * N, num_rels, L)
            stacked_ho = stacked_ho.view(B * C * N, self.num_rels, L)  # (B*C*N, num_rels, L)
            # 卷积操作
            conv_out = self.conv(stacked_ho)  # (B*C*N, output_dim, L)
            # Reshape back to (B, C, output_dim, N, L)
            conv_out = conv_out.view(B, C, -1, N, L)  # (B, C, output_dim, N, L)
            # 平均不同通道
            final_features = conv_out.mean(dim=1)  # (B, output_dim, N, L)

        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

        return final_features

# 使用示例

if __name__ == "__main__":
    # 定义参数
    B = 2  # 批次数
    C_out = 16  # 输出特征维度
    gdepth = 2  # 传播深度
    N = 6  # 节点数
    L = 4  # 特征长度
    num_rels = 5  # 关系类型数量

    # 初始化五种关系类型的特征张量
    ho = [torch.randn(B, C_out * (gdepth + 1), N, L) for _ in range(num_rels)]
    print("输入形状: ", ho[0].shape)

    # 选择聚合方法
    method = 'conv_fusion'  # 选择不同的方法：'sum', 'average', 'max', 'weighted_sum', 'attention', 'mlp', 'conv_fusion'

    # 初始化聚合器
    if method == 'attention':
        aggregator = FeatureAggregator(
            method=method,
            num_rels=num_rels,
            embed_size=C_out * (gdepth + 1)
        )
    elif method == 'mlp':
        aggregator = FeatureAggregator(
            method=method,
            num_rels=num_rels,
            hidden_dim=128,
            output_dim=C_out * (gdepth + 1)
        )
    elif method == 'conv_fusion':
        aggregator = FeatureAggregator(
            method=method,
            num_rels=num_rels,
            output_dim=C_out * (gdepth + 1)
        )
    else:
        aggregator = FeatureAggregator(
            method=method,
            num_rels=num_rels
        )

    # 前向传播
    final_features = aggregator(ho)

    print(f"聚合方法: {method}")
    print("输出特征矩阵的形状:", final_features.shape)
    # print("输出特征矩阵:")
    # print(final_features)