import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Data, Batch
from ogb.utils import smiles2graph
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
path = './data/Cell/cell_feature.csv'
def spilt_gene(gene_exp, device):
    spilt_cell_lines = pd.read_csv('./data/split_cell_lines.csv')
    keys = spilt_cell_lines['classify'].unique().tolist()
    #gene_exp = pd.read_csv(path, index_col=0)
    group_dict = spilt_cell_lines.set_index('genes')['classify'].to_dict()
    # 按组划分列
    grouped_columns = {}
    for col in gene_exp.columns:
        if col in group_dict:
            group_name = group_dict[col]
            grouped_columns.setdefault(group_name, []).append(col)
    # 生成分组后的子 DataFrame
    grouped_gene_exp = {group: gene_exp[cols] for group, cols in grouped_columns.items()}
    sub_gene = []
    for key in keys:
        sub_gene.append(torch.tensor(grouped_gene_exp[key].values.tolist(),dtype=torch.float32).to(device))

    return sub_gene

def split_drug_Smiles(drugs_smiles, skip, device):
    sub_smiles_list = []
    batch = []
    drug_cell_batch = []
    num = 0
    sub_num = 0
    edge_idxes, edge_feats, node_feats, lstnode, sub_batch, pe = [[],[]], [], [], 0, [], []
    for SMILES in drugs_smiles:
        try:
            mol = Chem.MolFromSmiles(SMILES)
            m = list(BRICS.BRICSDecompose(mol, returnMols=True))
            batch.extend([num]*len(m))
            drug_cell_batch.extend([num]*(len(m) + 13))
            for s in m:
                x, edge_index, edge_attr, p = smiles_to_graph(s, skip)
                node_feats.extend(x)
                pe.extend(p)
                edge_index[0] = [x + lstnode for x in edge_index[0]]
                edge_idxes[0].extend(edge_index[0])
                edge_index[1] = [x + lstnode for x in edge_index[1]]
                edge_idxes[1].extend(edge_index[1])
                edge_feats.extend(edge_attr)
                sub_batch.extend([sub_num] * len(x))
                lstnode += len(x)
                sub_num += 1
            num += 1
        except:
            pass
    batch = torch.tensor(np.array(batch), dtype=torch.long).to(device)
    drug_cell_batch = torch.tensor(np.array(drug_cell_batch), dtype=torch.long).to(device)
    # for s in sub_smiles_list:
    #     x, edge_index, edge_attr = smiles_to_graph(s)
    #     node_feats.extend(x)
    #     edge_index[0] = [x + lstnode for x in edge_index[0]]
    #     edge_idxes[0].extend(edge_index[0])
    #     edge_index[1] = [x + lstnode for x in edge_index[1]]
    #     edge_idxes[1].extend(edge_index[1])
    #     edge_feats.extend(edge_attr)
    #     sub_batch.extend([num] * len(x))
    #     lstnode += len(x)
    #     num += 1


    x = torch.tensor(np.array(node_feats), dtype=torch.long).to(device)
    pe = torch.tensor(np.array(pe), dtype=torch.float).to(device)
    edge_index = torch.tensor(np.array(edge_idxes), dtype=torch.long).to(device)
    edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.long).to(device)
    sub_batch = torch.tensor(np.array(sub_batch), dtype=torch.long).to(device)
    sub_drug_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pe=pe, batch=sub_batch)

    return sub_drug_data, batch, drug_cell_batch

def smiles_to_graph(mol, skip):
    #mol = Chem.MolFromSmiles(mol)
    if mol is None:  # 无效SMILES处理
        return None

    # 原子特征（这里使用6维简化特征）
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))

    # 边索引（键）
    edges = [[],[]]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges[0].append(i)
        edges[1].append(j)  # 无向图需双向
        edges[0].append(j)
        edges[1].append(i)  # 无向图需双向

    # 节点特征
    x = atom_features

    # 可选：边特征（如键类型）
    edge_attr = []
    for bond in mol.GetBonds():
        # 边特征 (3维)
        bond_feat = bond_to_feature_vector(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)  # 双向
    # edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    num_nodes = len(x)
    edge_index = np.array(edges)

    # 创建邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # 计算RWSE
    rwse = compute_rwse(adj_matrix, skip)

    return x, edges, edge_attr, rwse


def compute_rwse(adj_matrix: np.ndarray, L: int = 10) -> np.ndarray:
    """
    计算单个图的随机游走结构编码(RWSE)

    参数:
        adj_matrix (np.ndarray): 图的邻接矩阵 (N x N)
        L (int): 随机游走的最大步长

    返回:
        np.ndarray: RWSE位置编码矩阵 (N x L)
    """
    # 添加自环避免孤立节点问题
    np.fill_diagonal(adj_matrix, 1)

    # 计算度矩阵
    deg = np.sum(adj_matrix, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    # 计算归一化邻接矩阵
    A_hat = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # 计算随机游走概率矩阵
    RW = np.eye(adj_matrix.shape[0])  # 从单位矩阵开始
    rwse_list = [np.diag(RW).copy()]  # 存储每一步的对角线元素

    # 迭代计算L步
    for _ in range(L - 1):
        RW = RW @ A_hat
        rwse_list.append(np.diag(RW).copy())

    # 组合结果 (N x L)
    return np.stack(rwse_list, axis=1)


def add_rwse_to_graphs(graphs: list, L: int = 10) -> list:
    """
    为多个图添加RWSE位置编码

    参数:
        graphs (list): PyG Data对象的列表
        L (int): 随机游走的最大步长

    返回:
        list: 添加了RWSE编码的PyG图列表
    """
    graphs_with_rwse = []

    for graph in graphs:
        # 获取邻接矩阵
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index.numpy()

        # 创建邻接矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1

        # 计算RWSE
        rwse = compute_rwse(adj_matrix, L)

        # 转换为Tensor并添加到图
        graph.rwse = torch.tensor(rwse, dtype=torch.float)
        graphs_with_rwse.append(graph)

    return graphs_with_rwse



def get_select_cell_drug_data(basis_drug_list, Drug_cell_response):
    drug_list_lower = [drug.strip().lower() for drug in basis_drug_list]
    data = Drug_cell_response[Drug_cell_response['DRUG_NAME'].str.strip().str.lower().isin(drug_list_lower)]
    return data

def get_select_patient_drug_data(basis_drug_list, Drug_patient_response):
    data = Drug_patient_response[Drug_patient_response['DRUG_NAME'].str.strip().isin(basis_drug_list)]
    return data

def get_response_data(path):
    response_data = pd.read_csv(path)
    return response_data

class MyDataset(Dataset):
    def __init__(self, dataset):
        super(MyDataset, self).__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][3]

def getDataload(dataset, batch_size):
    dataset = MyDataset(dataset)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0)
    return dataset


# 核心划分函数
def stratified_drug_split(df, drugs, test_size=0.4, random_state=42):
    train_list, test_list = [], []

    for drug in drugs:
        drug_df = df[df['DRUG_NAME'] == drug]

        if len(drug_df) == 0:
            continue

        # 分层抽样（按response）
        stratify_col = drug_df['response']
        train_drug, test_drug = train_test_split(
            drug_df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )

        train_list.extend(train_drug.values.tolist())
        test_list.extend(test_drug.values.tolist())
    return train_list, test_list



def get_fine_tuning_Dataload(dataset, drug_list, batch_size):
    train_data, test_data = stratified_drug_split(dataset, drug_list)
    train_dataset = MyDataset(train_data)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataset = MyDataset(test_data)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataset, test_dataset

def auc_aupr_by_drugs(drug_list, patient_list, label, pred):
    # 示例数据结构 (替换为你的实际数据)
    # df需要包含三列：药物名称、真实标签、预测概率
    data = {
        'drug_name': drug_list,
        'patient_name' : patient_list,
        'true_label': label,
        'pred_prob': pred
    }
    df = pd.DataFrame(data)

    # 存储结果的字典
    results = {
        'drug_name': [],
        'auc': [],
        'aupr': []
    }

    # 分组计算
    for name, group in df.groupby('drug_name'):
        y_true = group['true_label'].values
        y_prob = group['pred_prob'].values

        # 跳过单一类别样本组
        if len(np.unique(y_true)) < 2:
            print(f"警告: {name} 组只有单一类别样本，跳过计算")
            continue

        # 计算AUC
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_score = np.nan

        # 计算AUPR (PR-AUC)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        aupr_score = auc(recall, precision)

        # 存储结果
        results['drug_name'].append(name)
        results['auc'].append(auc_score)
        results['aupr'].append(aupr_score)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


# spilt_gene(path)