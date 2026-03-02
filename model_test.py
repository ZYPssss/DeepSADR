import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set


# 模拟分子图数据（这里只是简单示例，真实场景需要根据具体分子数据格式来构建）
# 假设我们有100个分子图数据，每个分子图有节点特征维度为5，边连接关系等
num_molecules = 100
num_node_features = 5
num_classes = 1  # 假设预测一个分子性质数值，所以输出维度为1（类似回归任务）

# 构建分子图数据列表（在实际中，要从分子文件等解析并构建这些图数据对象）
molecular_graphs = []
for _ in range(num_molecules):
    # 随机生成节点特征矩阵（节点数量、节点特征维度），这里假设每个分子有不同数量节点，示例中简单随机范围
    num_nodes = torch.randint(5, 15, (1,)).item()  # 节点数量在5到15之间
    x = torch.randn(num_nodes, num_node_features)
    # 随机生成边索引（2, 边数量），这里简单模拟边连接情况，真实要根据分子结构确定
    edge_index = torch.randint(0, num_nodes, (2, torch.randint(10, 20, (1,)).item()))
    y = torch.randn(1)  # 模拟分子性质标签（一个数值）
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    molecular_graphs.append(graph_data)

# 设置batch_size
batch_size = 10
data_loader = DataLoader(molecular_graphs, batch_size=batch_size, shuffle=True)

# 定义基于GCN的分子性质预测模型
class MolecularGCNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MolecularGCNPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = global_max_pool
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = nn.functional.relu(x)
        x =   global_max_pool(x, batch)# 对节点特征取平均池化（一种简单聚合方式）
        x = self.fc(x)
        return x


model = MolecularGCNPredictor(num_node_features, 16, num_classes)

# 定义损失函数，使用均方误差损失用于回归任务
criterion = nn.MSELoss()
# 定义优化器，示例使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randint(5, 3)
index = [0, 0, 1, 1, 1]


for epoch in range(100):  # 训练轮次，可按需调整
    running_loss = 0.0
    for batch_data in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data.y.unsqueeze(1))  # 要调整标签维度匹配输出维度
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f'Epoch {epoch + 1} Loss: {running_loss / len(data_loader)}')