import torch
import torch.nn.functional as F
import torch.nn as nn
class cell_sub_feat(nn.Module):
    def __init__(self, input_dim_gene, output_dim_gene):
        super(cell_sub_feat, self).__init__()
        mlp_hidden_dims_gene = [256, 64]
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [output_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        v = v.float()
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v