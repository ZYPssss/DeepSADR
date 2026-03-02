import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_add_pool, global_mean_pool


class GNNPlusLayer(MessagePassing):
    """GNN⁺核心层，支持GCN/GIN/GatedGCN变体"""

    def __init__(self, gnn_type, in_dim, out_dim, use_edge_feat,
                 use_norm, use_dropout, use_residual, use_ffn,
                 dropout_rate=0.1, eps=0.0):
        super().__init__(aggr='add')
        self.gnn_type = gnn_type
        self.use_edge_feat = use_edge_feat
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.use_ffn = use_ffn

        # 消息传递参数
        if gnn_type == 'GCN':
            self.lin = nn.Linear(in_dim, out_dim)
            if use_edge_feat:
                self.edge_lin = nn.Linear(3, out_dim)

        elif gnn_type == 'GIN':
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim)
            )
            self.eps = eps

        elif gnn_type == 'GatedGCN':
            self.lin = nn.Linear(in_dim, out_dim)
            self.gate_lin = nn.Linear(2 * in_dim, 1)

        # 归一化层
        if use_norm:
            self.norm = nn.BatchNorm1d(out_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None

        # FFN模块
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(out_dim, 2 * out_dim),
                nn.ReLU(),
                nn.Linear(2 * out_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x, edge_index, edge_attr=None):
        # 残差连接保留输入
        residual = x

        # 消息传递
        if self.gnn_type == 'GCN':
            # 计算归一化系数
            row, col = edge_index
            deg = degree(col, x.size(0)).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # 消息传递
            out = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)
            out = self.lin(out)

        elif self.gnn_type == 'GIN':
            out = self.propagate(edge_index, x=x)
            out = self.mlp((1 + self.eps) * x + out)

        elif self.gnn_type == 'GatedGCN':
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 归一化
        if self.use_norm:
            out = self.norm(out)

        # 激活函数
        out = torch.relu(out)

        # Dropout
        if self.use_dropout:
            out = self.dropout(out)

        # 残差连接
        if self.use_residual:
            out = out + residual

        # FFN处理
        if self.use_ffn:
            out = self.ffn(out) + out  # FFN残差连接

        return out

    def message(self, x_j, norm, edge_attr=None):
        msg = x_j
        if self.gnn_type == 'GCN':
            msg = norm.view(-1, 1) * x_j
            if self.use_edge_feat and edge_attr is not None:
                msg += self.edge_lin(edge_attr)
        return msg


class GNNPlus(nn.Module):
    """完整GNN⁺架构"""

    def __init__(self, args):
        super().__init__()
        self.use_pe = args.use_pe
        in_dim = args.s_in_dim
        out_dim = args.s_out_dim
        hidden_dim = args.s_hidden_dim
        pe_dim = args.skip
        num_layers = args.num_layers
        gnn_type = args.gnn_type
        use_edge_feat = args.use_edge_feat
        use_norm = args.use_norm
        use_dropout = args.use_dropout
        use_residual = args.use_residual
        use_ffn = args.use_ffn
        dropout_rate = args.dropout

        # 位置编码处理
        if args.use_pe:
            self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
            in_dim = in_dim + hidden_dim

        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GNN⁺层堆叠
        self.layers = nn.ModuleList([
            GNNPlusLayer(gnn_type, hidden_dim, hidden_dim, use_edge_feat,
                 use_norm, use_dropout, use_residual, use_ffn,
                 dropout_rate)
            for _ in range(num_layers)
        ])

        self.pool = global_mean_pool

        # 图级读出函数
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 位置编码融合
        if self.use_pe:
            x = torch.cat([x, self.pe_encoder(data.pe)], dim=-1)

        x = self.input_proj(x.float())

        # 消息传递
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr.float())

        # 图级读出（全局平均池化）
        graph_emb = self.pool(x, data.batch)

        # 预测头
        return self.readout(graph_emb)

