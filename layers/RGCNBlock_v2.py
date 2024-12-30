import torch  
import torch.nn as nn  
import torch.nn.functional as F  


# class LinearTransform(nn.Module):  
#     """  
#     使用二维卷积层（1x1卷积核）实现线性变换的类。  
#  
#     将输入张量从形状 (batch_size, c_out * (gdepth +1), num_nodes, l)  
#     转换为形状 (batch_size, c_out, num_nodes, l)。  
#  
#     参数:  
#         in_channels (int): 输入通道数，即 c_out * (gdepth +1)。  
#         out_channels (int): 输出通道数，即 c_out。  
#     """  
#     def __init__(self, in_channels, out_channels):  
#         super(LinearTransform, self).__init__()  
#         self.conv = nn.Conv2d(  
#             in_channels=in_channels,  
#             out_channels=out_channels,  
#             kernel_size=1,  # 1x1卷积核  
#             stride=1,  
#             padding=0,  
#             bias=True  # 根据需要选择是否使用偏置  
#         )  
#  
#     def forward(self, ho):  
#         """  
#         前向传播。  
#  
#         参数:  
#             ho (torch.Tensor): 输入张量，形状为 (batch_size, c_out * (gdepth +1), num_nodes, l)。  
#  
#         返回:  
#             torch.Tensor: 输出张量，形状为 (batch_size, c_out, num_nodes, l)。  
#         """  
#         return self.conv(ho)  


class FeatureAggregator(nn.Module):  
    """  
    特征聚合模块，支持多种聚合方法。  

    支持的聚合方法包括：  
        - 'sum': 元素级求和  
        - 'average': 元素级平均  
        - 'max': 元素级最大值  
        - 'weighted_sum': 可学习的加权求和  
        - 'attention': 注意力机制  
        - 'mlp': 多层感知机  
        - 'conv_fusion': 卷积融合  

    参数:  
        method (str): 聚合方法的名称。  
        num_rels (int): 关系类型的数量。  
        embed_size (int, optional): 注意力机制或其他需要的嵌入尺寸。  
        hidden_dim (int, optional): MLP的隐藏层尺寸。  
        output_dim (int, optional): MLP的输出尺寸。  
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
            raise ValueError(f"不支持的聚合方法: {self.method}")  

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

            # 重排列为 (B, N, L, num_rels, C)  
            stacked_ho = stacked_ho.permute(0, 3, 4, 1, 2).contiguous()  # (B, N, L, num_rels, C)  
            # 重塑为 (B * N * L, num_rels, C)  
            ho_reshape = stacked_ho.view(B * N * L, self.num_rels, C)  # (B*N*L, num_rels, C)  

            # 计算注意力分数  
            attn_scores = self.attention(ho_reshape)  # 形状: (B*N*L, num_rels, 1)  
            attn_scores = attn_scores.squeeze(-1)  # (B*N*L, num_rels)  
            attn_weights = F.softmax(attn_scores, dim=1)  # (B*N*L, num_rels)  

            # 应用注意力权重  
            attn_weights = attn_weights.unsqueeze(-1)  # (B*N*L, num_rels, 1)  
            weighted_features = ho_reshape * attn_weights  # (B*N*L, num_rels, C)  
            final_features = weighted_features.sum(dim=1)  # (B*N*L, C)  

            # 重塑回 (B, C, N, L)  
            final_features = final_features.view(B, N, L, C).permute(0, 3, 1, 2).contiguous()  # (B, C, N, L)  

        elif self.method == 'mlp':  
            # MLP 聚合  
            B, C, N, L = ho[0].shape  
            stacked_ho = torch.stack(ho, dim=1)  # (B, num_rels, C, N, L)  
            # 重塑为 (B, num_rels, C * N * L)  
            ho_reshape = stacked_ho.view(B, self.num_rels, -1)  # (B, num_rels, C*N*L)  
            # 展平 num_rels 和 C*N*L 以作为 MLP 输入  
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
            # 重塑为 (B * C * N, num_rels, L)  
            stacked_ho = stacked_ho.view(B * C * N, self.num_rels, L)  # (B*C*N, num_rels, L)  
            # 卷积操作  
            conv_out = self.conv(stacked_ho)  # (B*C*N, output_dim, L)  
            # 重塑回 (B, C, output_dim, N, L)  
            conv_out = conv_out.view(B, C, -1, N, L)  # (B, C, output_dim, N, L)  
            # 平均不同通道  
            final_features = conv_out.mean(dim=1)  # (B, output_dim, N, L)  

        else:  
            raise ValueError(f"不支持的聚合方法: {self.method}")  

        return final_features  


class linear(nn.Module):  
    """  
    定义一个名为linear的类，继承自nn.Module，用于实现一个线性层。  
    """  

    def __init__(self, c_in, c_out, bias=True):  
        """  
        构造函数，初始化线性层。  

        参数:  
            c_in (int): 输入通道数。  
            c_out (int): 输出通道数。  
            bias (bool, optional): 是否使用偏置。默认True。  
        """  
        super(linear, self).__init__()  
        # 使用1x1卷积实现线性变换  
        self.mlp = torch.nn.Conv2d(  
            in_channels=c_in,  
            out_channels=c_out,  
            kernel_size=(1, 1),  
            padding=(0, 0),  
            stride=(1, 1),  
            bias=bias  
        )  

    def forward(self, x):  
        """  
        前向传播函数。  

        参数:  
            x (torch.Tensor): 输入张量。  

        返回:  
            torch.Tensor: 线性变换后的输出。  
        """  
        return self.mlp(x)  


class nconv(nn.Module):  
    """  
    定义一个名为nconv的类，继承自nn.Module，用于图卷积操作。  
    """  

    def __init__(self):  
        """  
        构造函数，初始化nconv模块。  
        """  
        super(nconv, self).__init__()  

    def forward(self, x, A):  
        """  
        前向传播函数，执行图卷积操作。  

        参数:  
            x (torch.Tensor): 输入张量，形状为 (batch_size, c, w, l)。  
            A (torch.Tensor): 邻接矩阵，形状为 (v, w)。  

        返回:  
            torch.Tensor: 图卷积后的输出张量，形状为 (batch_size, c, v, l)。  
        """  
        # 使用爱因斯坦求和约定进行张量乘法  
        # 'ncwl,vw->ncvl' 表示：  
        # n: 批量大小  
        # c: 通道数  
        # w: 输入张量的宽度，与邻接矩阵的行数对应  
        # l: 输入张量的长度  
        # v: 邻接矩阵的列数，表示输出张量的宽度  
        x = torch.einsum('ncwl,vw->ncvl', (x, A))  
        return x.contiguous()  


# class RGCNConv(nn.Module):  
#     def __init__(self, in_features, out_features, num_relations, activation=None):  
#         super(RGCNConv, self).__init__()  
#         self.num_relations = num_relations  
#         self.in_features = in_features  
#         self.out_features = out_features  
#         self.weight = nn.Parameter(torch.Tensor(num_relations, in_features, out_features))  
#         self.activation = activation  
#         nn.init.xavier_uniform_(self.weight.data, gain=1.414)  
#  
#     def forward(self, x, adj):  
#         """  
#         前向传播函数。  
#  
#         参数:  
#             x (torch.Tensor): 输入张量，形状为 (batch_size, in_features, num_nodes, l)。  
#             adj (torch.Tensor): 邻接矩阵，形状为 (num_relations, num_nodes, num_nodes)。  
#  
#         返回:  
#             torch.Tensor: 输出张量，形状为 (batch_size, out_features, num_nodes, l)。  
#         """  
#         out = torch.zeros(  
#             x.size(0),  
#             self.out_features,  
#             x.size(2),  
#             x.size(3),  
#             device=x.device  
#         )  
#  
#         for r in range(self.num_relations):  
#             H_adj = torch.einsum('bicl, kj -> bijl', x, adj[r])  
#             H_trans = torch.einsum('bijl, io -> bojl', H_adj, self.weight[r])  
#             out = out + H_trans  
#  
#         if self.activation:  
#             out = self.activation(out)  
#  
#         return out  


class MixPropRGCN(nn.Module):  
    """  
    混合传播的关系图卷积网络（MixPropRGCN）模块。  

    参数:  
        c_in (int): 输入通道数。  
        c_out (int): 输出通道数。  
        gdepth (int): 图传播的深度。  
        dropout (float): Dropout比率。  
        alpha (float): 融合系数。  
        num_rel (int): 关系类型的数量。  
        agg_method (str): 聚合方法名称。  
        activation (callable, optional): 激活函数。默认使用F.gelu。  
    """  

    def __init__(self, c_in, c_out, gdepth, dropout, alpha, num_rel, agg_method, activation=F.gelu):  
        super(MixPropRGCN, self).__init__()  
        self.gdepth = gdepth  
        self.alpha = alpha  
        self.dropout = nn.Dropout(dropout)  
        self.activation = activation  

        # 使用nconv进行图卷积操作  
        self.rgcn_layers = nconv()  
        # 线性层，用于融合不同传播深度的特征  
        self.mlp = linear((gdepth + 1) * c_in, c_out)  
        # 特征聚合模块  
        self.agg = FeatureAggregator(  
            method=agg_method,  
            num_rels=num_rel,  
            hidden_dim=64,  
            output_dim=c_out * (gdepth + 1)  
        )  

    def forward(self, x, adj):  
        """  
        前向传播函数。  

        参数:  
            x (torch.Tensor): 输入特征张量，形状为 (batch_size, c_in, num_nodes, l)。  
            adj (torch.Tensor): 邻接矩阵，形状为 (num_rel, num_nodes, num_nodes)。  

        返回:  
            torch.Tensor: 输出特征张量，形状为 (batch_size, c_out, num_nodes, l)。  
        """  
        ho = []  
        for j in range(adj.size(0)):  
            # 添加自环  
            sub_adj = adj[j, :, :] + torch.eye(adj[j, :, :].size(1)).to(x.device)  
            d = sub_adj.sum(1)  # 度矩阵  
            h = x  
            out = [h]  
            a = sub_adj / d.view(-1, 1)  # 归一化邻接矩阵  

            for i in range(self.gdepth):  
                rgcn_output = self.rgcn_layers(h, a)  # 图卷积输出  
                h = self.alpha * x + (1 - self.alpha) * rgcn_output  # 融合原始输入和图卷积输出  
                h = self.dropout(h)  # 应用Dropout  
                out.append(h)  

            # 拼接所有传播深度的输出  
            ho.append(torch.cat(out, dim=1))  # (batch_size, c_out * (gdepth +1), num_nodes, l)  

        # 聚合不同关系类型的特征  
        ho = self.agg(ho)  # (batch_size, c_out * (gdepth +1), num_nodes, l)  
        # 通过线性层降维  
        ho = self.mlp(ho)  # (batch_size, c_out, num_nodes, l)  
        return ho  


class GraphBlockRGCN(nn.Module):  
    """  
    图块关系图卷积网络（GraphBlockRGCN）模块。  

    参数:  
        c_out (int): 输出通道数。  
        d_model (int): 模型维度。  
        conv_channel (int): 卷积通道数。  
        skip_channel (int): 跳跃连接的通道数。  
        num_relations (int): 关系类型的数量。  
        gcn_depth (int): 图卷积的深度。  
        dropout (float): Dropout比率。  
        propalpha (float): 融合系数。  
        seq_len (int): 序列长度。  
        node_dim (int): 节点维度。  
        agg_method (str): 聚合方法名称。  
    """  

    def __init__(self, c_out, d_model, conv_channel, skip_channel,  
                 num_relations, gcn_depth, dropout, propalpha, seq_len, node_dim, agg_method):  
        super(GraphBlockRGCN, self).__init__()  

        # 初始卷积层，调整输入形状  
        self.start_conv = nn.Conv2d(  
            in_channels=1,  
            out_channels=conv_channel,  
            kernel_size=(d_model - c_out + 1, 1)  
        )  

        # 混合传播的关系图卷积网络模块  
        self.gconv1 = MixPropRGCN(  
            c_in=conv_channel,  
            c_out=skip_channel,  
            gdepth=gcn_depth,  
            dropout=dropout,  
            alpha=propalpha,  
            num_rel=num_relations,  
            agg_method=agg_method  
        )  

        self.gelu = nn.GELU()  

        # 结束卷积层  
        self.end_conv = nn.Conv2d(  
            in_channels=skip_channel,  
            out_channels=seq_len,  
            kernel_size=(1, seq_len)  
        )  

        # 线性层和归一化层  
        self.linear = nn.Linear(c_out, d_model)  
        self.norm = nn.LayerNorm(d_model)  

        # 可学习的邻接矩阵参数  
        self.adj_matrices = nn.Parameter(torch.abs(torch.randn(num_relations, c_out, c_out)))  

    def forward(self, x):  
        """  
        前向传播函数。  

        参数:  
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_nodes, d_model)。  

        返回:  
            torch.Tensor: 输出张量，形状为 (batch_size, num_nodes, d_model)。  
        """  
        # 计算邻接矩阵  
        adj = F.softmax(self.adj_matrices, dim=-1)  # (num_relations, num_nodes, num_nodes)  

        # 调整输入形状以适应Conv2d  
        out = x.unsqueeze(1).transpose(2, 3)  # (B, 1, d_model, T)  
        out = self.start_conv(out)  # (B, conv_channel, num_nodes, T)  

        # 图卷积操作  
        out = self.gconv1(out, adj)  # (B, skip_channel, num_nodes, T)  
        out = self.gelu(out)  

        # 结束卷积  
        out = self.end_conv(out).squeeze()  # (B, skip_channel, num_nodes, T)  

        # 应用线性层  
        out = self.linear(out)  # (B, num_nodes, d_model)  

        # 残差连接和归一化  
        return self.norm(x + out)  # (B, num_nodes, d_model)  


class RGCNModel(nn.Module):  
    """  
    关系图卷积网络（RGCN）模型。  

    参数:  
        num_relations (int): 关系类型的数量。  
        c_out (int): 输出通道数。  
        d_model (int, optional): 模型维度。默认值为512。  
        seq_len (int, optional): 序列长度。默认值为12。  
        node_dim (int, optional): 节点维度。默认值为32。  
        num_layers (int, optional): 模型层数。默认值为2。  
        conv_channel (int, optional): 卷积通道数。默认值为64。  
        skip_channel (int, optional): 跳跃连接通道数。默认值为128。  
        gcn_depth (int, optional): 图卷积深度。默认值为1。  
        dropout (float, optional): Dropout比率。默认值为0.3。  
        propalpha (float, optional): 融合系数。默认值为0.5。  
        agg_method (str, optional): 聚合方法名称。默认值为'sum'。  
    """  

    def __init__(self, num_relations, c_out, d_model=512, seq_len=12, node_dim=32,  
                 num_layers=2, conv_channel=64, skip_channel=128, gcn_depth=1,  
                 dropout=0.3, propalpha=0.5, agg_method='sum'):  
        super(RGCNModel, self).__init__()  
        layers = []  
        for i in range(num_layers):  
            layers.append(GraphBlockRGCN(  
                c_out=c_out,  # 与 MixPropRGCN 中的 c_out 一致  
                d_model=d_model,  
                conv_channel=conv_channel,  
                skip_channel=skip_channel,  
                num_relations=num_relations,  
                gcn_depth=gcn_depth,  
                dropout=dropout,  
                propalpha=propalpha,  
                seq_len=seq_len,  
                node_dim=node_dim,  
                agg_method=agg_method  
            ))  
        self.layers = nn.ModuleList(layers)  
        self.output_layer = nn.Linear(d_model, d_model)  

    def forward(self, x):  
        """  
        前向传播函数。  

        参数:  
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_nodes, d_model)。  

        返回:  
            torch.Tensor: 输出张量，形状为 (batch_size, num_nodes, d_model)。  
        """  
        for layer in self.layers:  
            x = layer(x)  
        out = self.output_layer(x)  
        return out  


if __name__ == "__main__":  
    # 假设有以下参数  
    batch_size = 32  
    seq_len = 96  # 时间步长或序列长度  
    d_model = 64  
    c_out = 10  # 可以与 T 不同  
    num_relations = 5  

    # 创建示例输入  
    x = torch.randn(batch_size, seq_len, d_model)  # 输入特征  
    print("x.shape: ", x.shape)  # 输出: torch.Size([32, 96, 64])  

    # 实例化模型  
    model = RGCNModel(  
        num_relations=num_relations,  
        c_out=c_out,  
        d_model=d_model,  
        node_dim=32,  
        num_layers=2,  
        conv_channel=64,  
        skip_channel=64,  
        gcn_depth=1,  
        dropout=0.3,  
        propalpha=0.5,  
        seq_len=seq_len  
    )  # 传递序列长度参数  

    # 前向传播  
    output = model(x)  # 输出形状: (batch_size, num_nodes, d_model)  
    print(f"[Main] Output shape: {output.shape}")  # 应输出: torch.Size([32, 96, 64])