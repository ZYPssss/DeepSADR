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
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=1024)

parser.add_argument("--in_dim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--dim_output", type=int, default=64)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--thred", type=float, default=0.71)
parser.add_argument("--fu_thred", type=float, default=0.71)
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

def fune_tune_model(args, drug_name):
    # 创建低保真模型 (自适应读出)
    print("Initializing models...")
    patient_gene_exp = pd.read_csv(argsdict['data_path']+'Patient/patient_feature.csv', index_col=0)
    patient_drug_Smiles = pd.read_csv(argsdict['data_path']+'Patient/drugs_Smiles.csv', index_col=0)
    test_drugs = [drug_name]
    for d in test_drugs:
        low_fidelity_model = torch.load(
            './premodel/{}+{}+{}+{}/low_fidelity_model++({}).pth'.format(args.num_layers, args.dropout, args.num_heads,
                                                                         args.skip, args.thred), map_location=device)
        low_fidelity_model_no = torch.load(
            './premodel/{}+{}+{}+{}/low_fidelity_model++({}).pth'.format(args.num_layers, args.dropout, args.num_heads,
                                                                         args.skip, args.thred), map_location=device)
        test_drug = [d]
        patient_response_data = get_response_data(argsdict['data_path'] + 'Patient/DP_response.csv')
        patient_response_data = get_select_patient_drug_data(test_drug, patient_response_data)
        ftrain_data, ftest_data = get_fine_tuning_Dataload(patient_response_data, test_drug, args.batch)
        high_fidelity_model = TransferLearningModel(low_fidelity_model, args, freeze_gnn=True).to(device)
        optimizer_high = optim.Adam(high_fidelity_model.parameters(), lr=0.0001)
        # 高保真微调
        params = {
                 'num_layers': [args.num_layers],
                 'dropout': [args.dropout],
                 'num_heads': [args.num_heads],
                 'thred': [args.thred],
                 'skip': [args.skip],
                 'fune_lr': [0.0001]
                 }
        params = pd.DataFrame(params)
        if not os.path.exists('./result/{}'.format(d)):
            os.makedirs('./result/{}'.format(d))
        params.to_csv('./result/{}/param.csv'.format(d), index=False)
        print("Fine-tuning high-fidelity model...")
        b_auc = 0
        b_aupr = 0
        for epoch in range(args.epochs):
            loss = 0.0
            high_fidelity_model.train()
            for idx, data in enumerate(tqdm(ftrain_data, desc='Iteration')):
                p_name, d_name, label = data
                p_name = list(p_name)
                d_name = list(d_name)
                label = torch.tensor(label, dtype=torch.float32).to(device)
                # 提取目标行（保持列表顺序和重复项）
                batch_patient_gene_exp = patient_gene_exp.loc[patient_gene_exp.index.intersection(p_name).unique().tolist()]  # 先获取唯一索引
                batch_patient_gene_exp = batch_patient_gene_exp.loc[p_name]  # 按原列表顺序和重复次数排列
                batch_patient_drug_smiles = patient_drug_Smiles.loc[patient_drug_Smiles.index.intersection(d_name).unique().tolist()]  # 先获取唯一索引
                batch_patient_drug_smiles = batch_patient_drug_smiles.loc[d_name]['Ismiles']  # 按原列表顺序和重复次数排列

                patient_subs = spilt_gene(batch_patient_gene_exp, device)
                drug_subs, batch, drug_patient_batch = split_drug_Smiles(batch_patient_drug_smiles, args.skip, device)
                optimizer_high.zero_grad()
                low_fidelity_model_no.eval()
                with torch.no_grad():
                    _, _, _, z = low_fidelity_model_no(patient_subs.copy(), drug_subs, batch, drug_patient_batch)
                    # 获取图级嵌入 (自适应读出可训练)
                    l_g = low_fidelity_model_no.get_graph_embedding(z, drug_patient_batch)

                pred_high = high_fidelity_model(patient_subs, drug_subs, batch, drug_patient_batch, l_g.requires_grad_(True))
                loss = F.mse_loss(pred_high.squeeze(), label)
                loss.backward()
                optimizer_high.step()
                loss += loss.item()

            high_fidelity_model.eval()
            # high_fidelity_model.low_fidelity_model.thred = args.fu_thred
            # low_fidelity_model_no.thred = args.fu_thred
            with torch.no_grad():
                drug_list = []
                patient_list = []
                label_list = []
                pred_list = []
                for idx, data in enumerate(tqdm(ftest_data, desc='Iteration')):
                    p_name, d_name, label = data
                    p_name = list(p_name)
                    d_name = list(d_name)
                    # 提取目标行（保持列表顺序和重复项）
                    batch_patient_gene_exp = patient_gene_exp.loc[patient_gene_exp.index.intersection(p_name).unique().tolist()]  # 先获取唯一索引
                    batch_patient_gene_exp = batch_patient_gene_exp.loc[p_name]  # 按原列表顺序和重复次数排列
                    batch_patient_drug_smiles = patient_drug_Smiles.loc[patient_drug_Smiles.index.intersection(d_name).unique().tolist()]  # 先获取唯一索引
                    batch_patient_drug_smiles = batch_patient_drug_smiles.loc[d_name]['Ismiles']  # 按原列表顺序和重复次数排列

                    patient_subs = spilt_gene(batch_patient_gene_exp, device)
                    drug_subs, batch, drug_patient_batch = split_drug_Smiles(batch_patient_drug_smiles, args.skip, device)
                    _, _, _, z = low_fidelity_model_no(patient_subs.copy(), drug_subs, batch, drug_patient_batch)
                    # 获取图级嵌入 (自适应读出可训练)
                    l_g = low_fidelity_model_no.get_graph_embedding(z, drug_patient_batch)
                    pred_high = high_fidelity_model(patient_subs, drug_subs, batch, drug_patient_batch, l_g)
                    pred_result = pred_high.squeeze().tolist()
                    drug_list.extend(d_name)
                    patient_list.extend(p_name)
                    label_list.extend(label.tolist())
                    pred_list.extend(pred_result)
                result = auc_aupr_by_drugs(drug_list, patient_list, label_list, pred_list)
                d_result = result[result['drug_name'] == d]
                auc = d_result['auc'].values[0]
                aupr = d_result['aupr'].values[0]
                # 定义你想要创建的文件夹路径
                if((b_auc < auc) and (b_aupr < aupr)):
                    b_auc = auc
                    b_aupr = aupr
                    folder_path = './result/{}'.format(d)
                    # 检查文件夹是否存在，如果不存在则创建
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    result.to_csv(folder_path + '/result.csv', index=False)
                    result = {
                        'drug_name': drug_list,
                        'patient_name': patient_list,
                        'true_label': label,
                        'pred_prob': pred_list
                    }
                    result = pd.DataFrame(result)
                    result.to_csv(folder_path + '/data.csv', index=False)
                    torch.save(low_fidelity_model, folder_path + '/pre_model.pth')
                    torch.save(high_fidelity_model, folder_path + '/fune_tune_model.pth')

            #print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(high_loader):.4f}")

    print("Training completed!")

if __name__ == '__main__':
    test_drugs = ['Fluorouracil', 'Temozolomide', 'Gemcitabine', 'Cisplatin', 'Sorafenib', 'Sunitinib', 'Doxorubicin',
                  'Erlotinib', 'Oxaliplatin']
    drug_name = test_drugs[2]
    fune_tune_model(args, drug_name)


