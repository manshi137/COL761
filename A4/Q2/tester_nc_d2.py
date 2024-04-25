import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
import pandas as pd

class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, edge_index, x):
        h = self.conv1(x, edge_index)
        h = F.tanh(h)
        h = self.conv2(h, edge_index)
        return h

def test(g, model,output_file):
    test_mask = g.test_mask
    features = g.x
    labels = g.y
    # print(g.HL)
    logits = model(g.edge_index, features )
    probabilities = F.softmax(logits)
    # print(probabilities)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    roc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), F.softmax(logits[test_mask], dim=1).cpu().detach().numpy(), multi_class='ovo',average='macro')
    print('Test accuracy:', test_acc.item())
    print('ROC AUC score:', roc)
    probs_numpy = probabilities.cpu().detach().numpy()
    pd.DataFrame(probs_numpy).to_csv(output_file, header=False, index=False)


model_path = sys.argv[1]
file_path = sys.argv[2]
output_path = sys.argv[3]
g = torch.load(file_path ,map_location='cpu')
num_nodes = g.num_nodes 

g.test_mask = torch.full((num_nodes,), True, dtype=torch.bool)

for i in range (num_nodes):
    if g.y[i]== -1: 
        g.test_mask[i] = False 
model = GCN(g.num_features, 8, 16)


model.load_state_dict(torch.load(model_path))
test(g, model,output_path)