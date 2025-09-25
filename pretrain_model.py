import pandas as pd
import argparse
from tqdm import tqdm
import random
import os
import os
import torch
import numpy as np
from DataLoader import spilt_gene, split_drug_Smiles, get_response_data, getDataload, get_select_cell_drug_data, get_fine_tuning_Dataload, get_select_patient_drug_data, auc_aupr_by_drugs
import torch.optim as optim
from model import *
from DeepModel.model import *

def set_seed(seed=42):
    # 基础设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description="cell-drug training script.")
parser.add_argument("--data-path", default='./data/')
parser.add_argument("--low-fidelity-label", default='SD', nargs='+')
parser.add_argument("--node-latent-dim", type=int, default=50)
parser.add_argument("--graph-latent-dim", type=int, default=256)
parser.add_argument("--cell_in_dim", type=list, default=[161, 122, 87, 166, 154, 80, 84, 56, 64, 152, 125, 86, 89])
parser.add_argument("--cell_out_dim", type=int, default=32)
parser.add_argument("--pre_epochs", type=int, default=200)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=1024)


parser.add_argument("--in_dim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--dim_output", type=int, default=64)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--thred", type=float, default=0.5)
parser.add_argument("--skip", type=int, default=10)


parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimensions (default: 300)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
parser.add_argument('--gnn_type', type=str, default="GCN")
parser.add_argument('--virtual_node', type=bool, default=False)
parser.add_argument('--residual', type=bool, default=False)


parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--s_in_dim", type=int, default=9)
parser.add_argument("--s_out_dim", type=int, default=32)
parser.add_argument("--s_hidden_dim", type=int, default=32)
parser.add_argument('--use_pe', type=bool, default=True)
parser.add_argument('--use_edge_feat', type=bool, default=True)
parser.add_argument('--use_norm', type=bool, default=True)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--use_ffn', type=bool, default=True)
parser.add_argument('--use_dropout', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.5)


# action = argparse.BooleanOptionalAction,

args = parser.parse_args()
argsdict = vars(args)

def pretrain_model(args):
    response_data = get_response_data(argsdict['data_path'] + 'Cell/DC_response.csv')
    basis_drug_list = ['5-Fluorouracil', 'Temozolomide', 'Gemcitabine', 'Cisplatin', 'Sorafenib', 'Sunitinib',
                       'Doxorubicin', 'Tamoxifen', 'Paclitaxel', 'Carmustine', 'Cetuximab', 'Methotrexate', 'Topotecan',
                       'Erlotinib', 'Irinotecan', 'Bicalutamide', 'Temsirolimus', 'Oxaliplatin', 'Docetaxel',
                       'Etoposide']
    response_data = get_select_cell_drug_data(basis_drug_list, response_data)
    gene_exp = pd.read_csv(argsdict['data_path'] + 'Cell/cell_feature.csv', index_col=0)
    drug_Smiles = pd.read_csv(argsdict['data_path'] + 'Cell/drugs_Smiles.csv', index_col=0)
    pre_train_data = getDataload(response_data.values.tolist(), args.batch)
    # 创建低保真模型 (自适应读出)
    print("Initializing models...")
    low_fidelity_model = SupervisedVGAE(args, readout_type='adaptive').to(device)
    # 创建优化器
    optimizer_low = optim.Adam(low_fidelity_model.parameters(), lr=0.00001)
    low_fidelity_model.train()
    min_loss = 9999999
    for epoch in range(args.pre_epochs):
        total_loss = 0.0
        for idx, data in enumerate(tqdm(pre_train_data, desc='Iteration')):
            cell_name, drug_name, label = data
            cell_name = list(cell_name)
            drug_name = list(drug_name)
            label = torch.tensor(label, dtype=torch.float32).to(device)
            # 提取目标行（保持列表顺序和重复项）
            batch_gene_exp = gene_exp.loc[gene_exp.index.intersection(cell_name).unique().tolist()]  # 先获取唯一索引
            batch_gene_exp = batch_gene_exp.loc[cell_name]  # 按原列表顺序和重复次数排列
            batch_drug_smiles = drug_Smiles.loc[drug_Smiles.index.intersection(drug_name).unique().tolist()]  # 先获取唯一索引
            batch_drug_smiles = batch_drug_smiles.loc[drug_name]['Ismiles']  # 按原列表顺序和重复次数排列

            cell_subs = spilt_gene(batch_gene_exp, device)
            drug_subs, batch, drug_cell_batch = split_drug_Smiles(batch_drug_smiles, args.skip, device)

            pred, mu, logvar, z = low_fidelity_model(cell_subs, drug_subs, batch, drug_cell_batch)

            loss = low_fidelity_model.loss(pred, label, mu, logvar, drug_cell_batch)
            loss.backward()
            optimizer_low.step()
            total_loss += loss.item()
        print(total_loss)
        if(total_loss < min_loss):
            min_loss = total_loss
            torch.save(low_fidelity_model, './premodel/{}+{}+{}+{}/low_fidelity_model++({}).pth'.format(args.num_layers, args.dropout, args.num_heads, args.skip, args.thred))

    print("Training completed!")

if __name__ == '__main__':
    h = args.thred
    # H = []
    # for i in range(9):
    #     H.append(0.01 * (i + 1) + h)
    H = [h+0.07, h+0.08, h+0.09]
    for thred in H:
        args.thred = thred
        pretrain_model(args)


