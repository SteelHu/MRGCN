import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class KGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, node_dim, dropout=0.2):
        super(KGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.num_relations = num_relations
        self.dropout = dropout
        # 定义节点向量1
        self.nodevec1 = nn.Parameter(torch.randn(out_channels, node_dim), requires_grad=True)
        # 定义节点向量2
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, out_channels), requires_grad=True)

        # Relation-specific transformation matrices
        self.relation_trans = nn.Embedding(num_relations, in_channels * out_channels)

        # Weight matrices for transforming node features
        self.weight = nn.Linear(in_channels, out_channels, bias=False)

        # Attention mechanism
        self.attention = nn.Parameter(torch.Tensor(out_channels, 1))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_type):
        """
        x: Node features [B, t, d_model]
        """
        # Transform node features
        x = self.weight(x)  # [d_model, out_channels]

        # Prepare adjacency in edge list format
        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adj = adj.tocoo()
        edge_index = torch.LongTensor([adj.row, adj.col])  # [2, E]
        edge_type = torch.LongTensor(edge_type)  # [E]

        # Get relation-specific transformation
        rel_trans = self.relation_trans(edge_type)  # [E, in_channels * out_channels]
        rel_trans = rel_trans.view(-1, self.in_channels, self.out_channels)  # [E, in_channels, out_channels]

        # Perform message passing
        messages = torch.bmm(x[edge_index[0]], rel_trans)  # [E, out_channels]

        # Aggregate messages
        out = torch.zeros(x.size(0), self.out_channels).to(x.device)
        out = out.index_add(0, edge_index[1], messages)

        # Apply attention
        attention_scores = torch.matmul(out, self.attention).squeeze(-1)  # [N]
        attention_weights = F.softmax(attention_scores, dim=0)  # [N]
        attention_weights = self.dropout_layer(attention_weights)

        out = out * attention_weights.unsqueeze(-1)  # [N, out_channels]

        # Non-linearity
        out = F.relu(out)

        return out

class KGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gcn_layers=2, dropout=0.2):
        super(KGATBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.gcn_layers = gcn_layers
        for _ in range(gcn_layers):
            self.convs.append(KGATConv(in_channels, out_channels, num_relations, dropout))
            in_channels = out_channels  # For subsequent layers
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, edge_type):
        for conv in self.convs:
            x = conv(x, adj, edge_type)
            x = self.layer_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x
