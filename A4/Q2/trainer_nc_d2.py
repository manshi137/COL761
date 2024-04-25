import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
# dataset = dgl.data.CoraGraphDataset()
# print('Number of categories:', dataset.num_classes)
# g = dataset[0]
# print('Node features')
# print(g.ndata)
# print('Edge features')
# print(g.edata)
# print("Train mask = ")
# print(g.ndata['train_mask'])

file_path = sys.argv[1]
output_path = sys.argv[2]
g = torch.load(file_path ,map_location='cpu')
num_nodes = g.num_nodes 
g.train_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.val_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.test_mask = torch.full((num_nodes,), False, dtype=torch.bool)
print(g.train_mask)
# Assuming g.y is your tensor
unique_values = torch.unique(g.y)
num_classes = len(unique_values) - 1
print(num_classes)
print(unique_values)
for i in range ((num_nodes*8)//10):
    if g.y[i]!= -1: 
        g.train_mask[i] = True 
for i in range ((num_nodes*8)//10 , (num_nodes*9)//10 ):
    if g.y[i]!= -1: 
        g.val_mask[i] = True 
for i in range ((num_nodes*9)//10  , num_nodes):
    if g.y[i]!= -1: 
        g.test_mask[i] = True 
# edges = (g.edge_index[0], g.edge_index[1])  # Assuming edge_index is a tuple of source and destination nodes
    
# gr = dgl.graph(edges)
# gr = dgl.add_self_loop(gr)



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

# Load your graph data, assuming you have edge_index and node features x
# g = torch.load('your_data.pt', map_location='cpu')

num_nodes = g.num_nodes 
g.train_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.val_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.test_mask = torch.full((num_nodes,), False, dtype=torch.bool)

unique_values = torch.unique(g.y)
num_classes = len(unique_values) - 1

for i in range((num_nodes * 8) // 10):
    if g.y[i] != -1: 
        g.train_mask[i] = True 
for i in range((num_nodes * 8) // 10, (num_nodes * 9) // 10):
    if g.y[i] != -1: 
        g.val_mask[i] = True 
for i in range((num_nodes * 9) // 10, num_nodes):
    if g.y[i] != -1: 
        g.test_mask[i] = True 

def train(model, g, num_epochs=1001, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.x
    labels = g.y
    train_mask = g.train_mask
    val_mask = g.val_mask
    test_mask = g.test_mask

    for e in range(num_epochs):
        # Forward pass
        logits = model(g.edge_index, features)

        # Compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            # Compute prediction
            pred = logits.argmax(dim=1)

            # Compute accuracy
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Compute ROC AUC
            roc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), 
                                F.softmax(logits[test_mask], dim=1).cpu().detach().numpy(), 
                                multi_class='ovo')

            # Update best validation and test accuracies
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            print('Epoch {}, Loss: {:.3f}, Train Acc: {:.3f}, Val Acc: {:.3f} (Best {:.3f}), Test Acc: {:.3f} (Best {:.3f}), ROC Test: {:.3f}'
                  .format(e, loss.item(), train_acc.item(), val_acc.item(), best_val_acc.item(), test_acc.item(), best_test_acc.item(), roc))

# Instantiate the model
model = GCN(g.num_features, num_classes, 16)

# Train the model
train(model, g)
torch.save(model.state_dict(), output_path)


