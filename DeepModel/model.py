import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from DeepModel.set_transformer_models import *
from DeepModel.drug_sub_feat import GNN
from DeepModel.cell_sub_feat import cell_sub_feat
from DeepModel.GNNPlus import *
from torch.nn import Parameter
import time

#device = torch.device('cuda:0')

class SupervisedVGAE(nn.Module):

    def __init__(self, args, readout_type='adaptive'):
        super().__init__()
        cell_in_list = args.cell_in_dim
        self.thred = args.thred
        self.sub_cell_feat = nn.ModuleList([cell_sub_feat(cell_in, args.cell_out_dim) for cell_in in cell_in_list])
        #self.sub_feat_gnn = GNN(args)
        self.sub_feat_gnn = GNNPlus(args)
        self.weight = Parameter(torch.Tensor(args.emb_dim, args.cell_out_dim))
        self.glorot(self.weight)

        # 编码器
        self.conv1 = GCNConv(args.in_dim, args.hidden_dim)
        self.conv_mu = GCNConv(args.hidden_dim, args.latent_dim)
        self.conv_logvar = GCNConv(args.hidden_dim, args.latent_dim)

        # 读出函数
        self.readout_type = readout_type
        if readout_type == 'adaptive':
            self.readout = SetTransformer(args.latent_dim, args.num_heads, args.dim_output)
            self.readout_fn = self.adaptive_readout
        else:  # sum, mean, max
            self.readout_fn = {
                'sum': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool
            }[readout_type]

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def encode(self, x, edge_index, edge_attr):
        # 消息传递
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def adaptive_readout(self, z, batch):
        """处理批量图的自适应读出"""
        # 将节点特征按图分组
        unique_batches = torch.unique(batch)
        batch_embeddings = []

        for b in unique_batches:
            mask = (batch == b)
            # 为当前图的节点添加批次维度 [1, num_nodes, latent_dim]
            batch_nodes = z[mask].unsqueeze(0)
           # batch_nodes = z[mask]
            x = self.readout(batch_nodes)
            embedding =  torch.mean(x, dim=1)
            batch_embeddings.append(embedding)

        return torch.cat(batch_embeddings, dim=0)

    def get_graph_embedding(self, z, batch):
        """获取图级嵌入"""
        if self.readout_type == 'adaptive':
            return self.adaptive_readout(z, batch)
        else:
            return self.readout_fn(z, batch)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, cell_subs, drug_subs, batch, drug_cell_batch):
        for i, l in enumerate(self.sub_cell_feat):
            cell_subs[i] = l(cell_subs[i])

        sub_feat = self.sub_feat_gnn(drug_subs)
        unique_batches = torch.unique(batch)
        x = []
        edge_index1 = []
        edge_index2= []
        edge_attr = []
        graph_num = 0


        for b in unique_batches:
            mask = (batch == b)
            drug_sub_nodes = sub_feat[mask]
            cell_sub_nodes = []
            for i in range(len(cell_subs)):
                cell_sub_nodes.append(cell_subs[i][b].unsqueeze(0))
            cell_sub_nodes = torch.cat(cell_sub_nodes, dim=0)

            graph_weight = torch.matmul(drug_sub_nodes, torch.matmul(self.weight, cell_sub_nodes.T))
            graph_weight = torch.sigmoid(graph_weight)
            # 形状: [num_drug_sub + num_cell_sub, feature_dim]
            x.append(torch.cat([drug_sub_nodes, cell_sub_nodes], dim=0))

            sub_drug_indx, sub_cell_indx = torch.where(graph_weight >= self.thred)
            edge_weight = graph_weight[sub_drug_indx, sub_cell_indx]
            edge_attr.append(edge_weight)
            edge_attr.append(edge_weight)

            # 调整细胞系节点索引 (加偏移量)
            sub_cell_indx = sub_cell_indx + len(drug_sub_nodes)
            sub_cell_indx = sub_cell_indx + graph_num

            sub_drug_indx = sub_drug_indx + graph_num

            edge_index1.append(sub_drug_indx)
            edge_index1.append(sub_cell_indx)

            edge_index2.append(sub_cell_indx)
            edge_index2.append(sub_drug_indx)

            graph_num = graph_num + len(drug_sub_nodes) + len(cell_sub_nodes)


            # 增加维度适配PyG格式 [num_edges, 1]
        x = torch.cat(x, dim=0)
        edge_attr = torch.cat(edge_attr, dim=0)
        edge_attr = edge_attr.unsqueeze(1)
        edge_index1 = torch.cat(edge_index1, dim=0)
        edge_index2 = torch.cat(edge_index2, dim=0)
        edge_index = torch.stack([edge_index1, edge_index2], dim=0)

        # 编码
        mu, logvar = self.encode(x, edge_index, edge_attr)
        z = self.reparameterize(mu, logvar)

        # 图级嵌入
        h_g = self.get_graph_embedding(z, drug_cell_batch)


        # 属性预测
        pred = self.predictor(h_g)
        pred = torch.sigmoid(pred)
        return pred, mu, logvar, z

    def loss(self, pred, y, mu, logvar, data, lambda_rec=0.1, lambda_pred=1.0):
        # 重建损失 (邻接矩阵)
        # 注意: 实际实现中需要更复杂的重建计算，这里简化处理
        recon_loss = torch.tensor(0.0, device=pred.device)

        # KL散度
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # 预测损失
        pred_loss = F.mse_loss(pred.squeeze(), y)

        # 总损失
        return lambda_rec * (recon_loss + kl_loss) + lambda_pred * pred_loss


class TransferLearningModel(nn.Module):
    """迁移学习框架"""

    def __init__(self, low_fidelity_model, args, freeze_gnn=True):
        """
        Args:
            low_fidelity_model: 预训练的低保真模型
            freeze_gnn: 是否冻结GNN层，只训练读出层
        """
        super().__init__()
        self.low_fidelity_model = low_fidelity_model
        self.freeze_gnn = freeze_gnn

        # 冻结GNN层
        if freeze_gnn:
            # 首先冻结整个模型的所有参数
            for param in self.low_fidelity_model.parameters():
                param.requires_grad = False

            #self.low_fidelity_model.weight.requires_grad = True

            # 解冻读出函数部分（如果是自适应类型）
            if self.low_fidelity_model.readout_type == 'adaptive':
                for param in self.low_fidelity_model.readout.parameters():
                    param.requires_grad = True
            else:
                for param in self.low_fidelity_model.readout_fn.parameters():
                    param.requires_grad = True

            # for param in self.low_fidelity_model.conv1.parameters():
            #     param.requires_grad = False
            # for param in self.low_fidelity_model.conv_mu.parameters():
            #     param.requires_grad = False
            # for param in self.low_fidelity_model.conv_logvar.parameters():
            #     param.requires_grad = False

        # 新预测头 (针对高保真任务)
        self.predictor = nn.Sequential(
            nn.Linear(args.latent_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, patient_subs, drug_subs, batch, drug_patient_batch, l_g):
        # 使用预训练模型获取节点嵌入
        with torch.set_grad_enabled(not self.freeze_gnn):
            _, _, _, z = self.low_fidelity_model(patient_subs, drug_subs, batch, drug_patient_batch)
            # z = self.low_fidelity_model.reparameterize(mu, logvar)

        # 获取图级嵌入 (自适应读出可训练)
        h_g = self.low_fidelity_model.get_graph_embedding(z, drug_patient_batch)
        x = torch.cat([h_g, l_g], dim=1)

        # 高保真预测
        return torch.sigmoid(self.predictor(x))






