import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData, Batch


# class HeteroInception_Block_V1(nn.Module):
#     def __init__(self, metadata, hidden_channels, out_channels, num_nodes, num_kernels=6, init_weight=True):
#         super(HeteroInception_Block_V1, self).__init__()
#         self.metadata = metadata  # 包含节点类型和边类型的信息
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
#         self.num_nodes = num_nodes
#
#         # 为每种边类型定义不同的卷积核
#         self.convs = nn.ModuleList()
#         for i in range(self.num_kernels):
#             conv = HeteroConv({
#                 edge_type: SAGEConv(-1, out_channels) for edge_type in metadata[1]
#             }, aggr='mean')  # 可以选择 'sum', 'max' 等聚合方式
#             self.convs.append(conv)
#
#         if init_weight:
#             self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.convs.modules():
#             if isinstance(m, SAGEConv):
#                 nn.init.kaiming_normal_(m.lin_l.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.lin_l.bias, 0)
#                 nn.init.kaiming_normal_(m.lin_r.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.lin_r.bias, 0)
#
#     def forward(self, x, edge_index_dict):
#         res_list = []
#         for conv in self.convs:
#             out = conv(x, edge_index_dict)
#             res_list.append(out)
#
#             # 融合不同卷积核的输出
#         # 假设所有节点类型的输出具有相同的结构
#         aggregated = {}
#         for key in res_list[0].keys():
#             node_features = [res[key] for res in res_list]
#             aggregated[key] = torch.stack(node_features, dim=-1).mean(-1)
#
#         return aggregated


class LearnableAdjHeteroConv(nn.Module):
    def __init__(self,
                 node_types,  # dict: {type: list of node indices}
                 edge_types,  # list of tuples: (source_type, relation, target_type)
                 input_dim,  # 输入特征维度，例如 20
                 N,  # 节点数量
                 hidden_channels,
                 out_channels,
                 rank=16,
                 aggr='mean',
                 normalize=True,
                 add_self_loops=True):
        """
        初始化基于可学习邻接矩阵的异质图卷积模块。

        Args:
            node_types (dict): 节点类型字典，如 {'A': [0,1,3], 'B': [2,4,5,6], 'C': [7,8,9]}。
            edge_types (list of tuples): 边类型列表，每个边类型为 (源节点类型, 关系, 目标节点类型)。
            input_dim (int): 输入特征维度。
            hidden_channels (int): 输入到 SAGEConv 的隐藏层维度。
            out_channels (int): SAGEConv 的输出特征维度。
            rank (int): 低秩分解的秩，默认为16。
            aggr (str): 聚合方法，默认为'mean'。其他选项如'sum'，'max'。
            normalize (bool): 是否对邻接矩阵进行归一化，默认为True。
            add_self_loops (bool): 是否添加自环边，默认为True。
        """
        super(LearnableAdjHeteroConv, self).__init__()
        self.node_types = node_types
        self.original_edge_types = edge_types.copy()  # 保留原始边类型
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.rank = rank
        self.aggr = aggr
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.d_model = input_dim
        self.N = N
        # self.input_dim = input_dim  # 用于后续恢复特征维度

        self.linear1 = nn.Linear(self.d_model, N)
        self.linear2 = nn.Linear(self.d_model, self.out_channels)

        # 初始化可学习参数 U 和 V
        self.U = nn.ParameterDict()
        self.V = nn.ParameterDict()

        # 初始化卷积层：每个边类型都有一个 SAGEConv
        self.convs = nn.ModuleDict()

        for edge_type in edge_types:
            src, rel, dst = edge_type
            edge_type_str = '__'.join(edge_type)
            n_src = len(node_types[src])
            n_dst = len(node_types[dst])
            self.U[edge_type_str] = nn.Parameter(torch.Tensor(n_src, rank))
            self.V[edge_type_str] = nn.Parameter(torch.Tensor(n_dst, rank))
            nn.init.xavier_uniform_(self.U[edge_type_str])
            nn.init.xavier_uniform_(self.V[edge_type_str])
            # 这里将输入维度 -> hidden_channels
            self.convs[edge_type_str] = SAGEConv(input_dim, hidden_channels)

            # 如果需要添加自环边，初始化自环边类型的参数与SAGEConv
        if self.add_self_loops:
            self_edge_types = [(node_type, 'self', node_type) for node_type in node_types.keys()]
            for edge_type in self_edge_types:
                src, rel, dst = edge_type
                edge_type_str = '__'.join(edge_type)
                n_nodes = len(node_types[src])
                self.U[edge_type_str] = nn.Parameter(torch.Tensor(n_nodes, rank))
                self.V[edge_type_str] = nn.Parameter(torch.Tensor(n_nodes, rank))
                nn.init.xavier_uniform_(self.U[edge_type_str])
                nn.init.xavier_uniform_(self.V[edge_type_str])
                self.convs[edge_type_str] = SAGEConv(input_dim, hidden_channels)
                self.original_edge_types.append(edge_type)

                # 最终映射回到与输入相同的维度 (input_dim)
        self.final_linear = nn.Linear(hidden_channels, input_dim)

        # 定义激活函数用于邻接矩阵
        self.activation = nn.Sigmoid()

    def forward(self, x, node_types, edge_types, period):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入形状 [B, N, L_div_P, P]。
            node_types (dict): 节点类型 -> 对应节点在总维度中的索引。
            edge_types (list of tuples): (源节点类型, 关系, target_node_type)。
            period (int): 周期，用于 reshape 输出。

        Returns:
            torch.Tensor: 形状 [B, N, L_div_P, P] 的输出特征。
        """
        B, d_model, L_div_P, P = x.size()
        # 变换维度
        x = x.permute(0, 2, 3, 1).contiguous()
        # 执行linear1
        x = self.linear1(x)
        # 变换维度
        x = x.permute(0, 3, 1, 2).contiguous()

        device = x.device

        data_list = []
        for b in range(B):
            data = HeteroData()

            # 为每种节点类型提取对应的节点特征
            for node_type, indices in node_types.items():
                # 这里节点特征是 [num_nodes_type, L_div_P * P]
                node_features = x[b, indices, :, :].reshape(len(indices), -1)
                data[node_type].x = node_features

                # 构建每种边类型的边索引
            for edge_type in self.original_edge_types:
                src_type, rel, dst_type = edge_type
                n_src = len(node_types[src_type])
                n_dst = len(node_types[dst_type])

                src_nodes = torch.arange(n_src, dtype=torch.long)
                dst_nodes = torch.arange(n_dst, dtype=torch.long)
                src_repeat = src_nodes.repeat_interleave(n_dst)
                dst_tile = dst_nodes.repeat(n_src)
                edge_index = torch.stack([src_repeat, dst_tile], dim=0)  # [2, E]
                data[edge_type].edge_index = edge_index

            data_list.append(data)

            # 使用 Batch.from_data_list 建立批处理
        batch = Batch.from_data_list(data_list)

        # x_dict: 每种节点类型对应的特征
        x_dict = {}
        for node_type in node_types.keys():
            x_dict[node_type] = batch[node_type].x

            # edge_index_dict: 每种边类型对应的 edge_index
        edge_index_dict = {}
        for edge_type in self.original_edge_types:
            edge_index = batch[edge_type].edge_index
            edge_index_dict[edge_type] = edge_index

            # 构建 HeteroConv
        hetero_conv = HeteroConv({
            edge_type: self.convs['__'.join(edge_type)] for edge_type in edge_index_dict.keys()
        }, aggr=self.aggr)

        # 异质图卷积
        out_dict = hetero_conv(x_dict, edge_index_dict)

        # 先 ReLU，再将 hidden_channels → input_dim
        for key, val in out_dict.items():
            val = F.relu(val)
            val = self.final_linear(val)  # 映射回 input_dim
            out_dict[key] = val

            # 回写到 batch
        for node_type in node_types.keys():
            batch[node_type].x = out_dict[node_type]

            # 分离出每个样本
        out_data_list = batch.to_data_list()

        # 最终重构输出张量，形状仍为 [B, N, L_div_P, P]
        out = torch.zeros(B, d_model, L_div_P, P, device=device)
        for b in range(B):
            for node_type, indices in node_types.items():
                # 现在节点特征已回到 input_dim = L_div_P * P
                node_features = out_data_list[b][node_type].x.view(-1, L_div_P, P)
                out[b, indices, :, :] = node_features

        # 变换维度
        out = out.permute(0, 2, 3, 1).contiguous()
        # 执行linear2
        out = self.linear2(out)
        # 变换维度
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

    # 示例用法


if __name__ == "__main__":


    # 假设有3种节点类型：'A', 'B', 'C'
    node_types = {
        'A': [0, 1, 3],
        'B': [2, 4, 5, 6],
        'C': [7, 8, 9]
    }
    # 定义边类型，例如 ('A', 'connects', 'B'), ('B', 'connects', 'C')
    edge_types = [('A', 'connects', 'B'), ('B', 'connects', 'C')]

    # 每种节点类型的数量
    node_counts = {k: len(v) for k, v in node_types.items()}

    input_dim = 20  # 输入特征维度 (L_div_P * P)
    hidden_channels = 64  # 图卷积内部的隐藏维度
    out_channels = 128  # 此处未显式使用，如需要可后续扩展
    rank = 16
    period = 5
    length = 20  # 假设 length 不小于 period, 这里为 20
    N = 10  # 假设 N 为 10

    model = LearnableAdjHeteroConv(
        node_types=node_types,
        edge_types=edge_types,
        input_dim=input_dim,
        N=N,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        rank=rank,
        add_self_loops=True
    )

    B = 2  # batch size
    x = torch.randn(B, input_dim, length // period, period)  # [B, N, 4, 5] => input_dim=4*5=20

    print("Input shape:", x.shape)
    out = model(x, node_types, edge_types, period)
    print("Output shape:", out.shape)
    # 应为 [B, N, L_div_P, P]，即 [2, 10, 4, 5]
    # print(out)