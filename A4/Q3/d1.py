import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import itertools
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch import Tensor
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import negative_sampling
import sys
dataset_path = sys.argv[1]
model_save_path = sys.argv[2]
file_path = dataset_path
data = torch.load(file_path, map_location='cpu')


transform = T.Compose([
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
                      add_negative_train_samples=False),
])

data = transform(data)
# print(data)
train_data, val_data, test_data = data
print(train_data)
print(val_data)
print(test_data)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(train_data.x.size(1), 128, 64)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = train_data.negative_edges
    # print(neg_edge_index)
    edge_label_index = torch.cat(
        [train_data.positive_edges, neg_edge_index],
        dim=-1,
    )
    # print(edge_label)
    # print(edge_label_index)
    ones_tensor = torch.ones(train_data.positive_edges.size(1))
    zeros_tensor = torch.zeros(train_data.negative_edges.size(1))
    edge_label = torch.cat([
        ones_tensor,
        zeros_tensor
    ], dim=0)
    print("el = " , edge_label.size())
    out = model.decode(z, edge_label_index).view(-1)
    print("out  =  " ,out.size())
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 25):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)

torch.save(model.state_dict(), model_save_path)
