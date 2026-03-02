import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from rdkit import Chem
from rdkit.Chem import BRICS
device = torch.device('cuda:0')

model = torch.load('./result/Temozolomide/fune_tune_model.pth', map_location=device)
print('wehdeiuw')