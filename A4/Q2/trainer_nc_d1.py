#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sys
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
#from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
import networkx as nx
from utils.gen_HoHLaplacian import creat_L_SparseTensor
from sklearn.metrics import roc_auc_score
import random
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

file_path = sys.argv[1]
output_path = sys.argv[2]
g = torch.load(file_path ,map_location='cpu')
num_nodes = g.num_nodes 
g.train_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.val_mask = torch.full((num_nodes,), False, dtype=torch.bool)
g.test_mask = torch.full((num_nodes,), False, dtype=torch.bool)
unique_values = torch.unique(g.y)
num_classes = len(unique_values) - 1
print(num_classes)
random_permutation = torch.randperm(num_nodes)
for i in range ((num_nodes*8)//10):
    if g.y[random_permutation[i]]!= -1: 
        g.train_mask[random_permutation[i]] = True 
for i in range ((num_nodes*8)//10 , (num_nodes*9)//10 ):
    if g.y[random_permutation[i]]!= -1: 
        g.val_mask[random_permutation[i]] = True 
for i in range ((num_nodes*9)//10  , num_nodes):
    if g.y[random_permutation[i]]!= -1: 
        g.test_mask[random_permutation[i]] = True 
G = nx.Graph()
G.add_nodes_from(range(g.num_nodes))
G.add_edges_from(g.edge_index.numpy().transpose())
g.HL = creat_L_SparseTensor(G, maxCliqueSize=25 )

class HiGCN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, HL):

        hidden = x * (self.fW[0])
        for k in range(self.K):
            x = matmul(HL, x, reduce=self.aggr)
            gamma = self.fW[k + 1]
            hidden = hidden + gamma * x

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)


class HiGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes , hidden  , order , alpha , K , dprate, dropout):
        super(HiGCN, self).__init__()
        self.Order = order
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()

        for i in range(order):
            self.lin_in.append(Linear(num_features, hidden))
            self.hgc.append(HiGCN_prop(K, alpha, order))

        self.lin_out = Linear(hidden * order, num_classes)


        self.dprate = dprate
        self.dropout = dropout

    # def reset_parameters(self):
    #     self.hgc.reset_parameters()

    def forward(self, data):

        x, HL = data.x, data.HL
        x_concat = torch.tensor([])
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = self.lin_out(x_concat)
        # x_concat = F.leaky_relu(x_concat)

        return F.log_softmax(x_concat, dim=1)



# # Create t
# he model with given dimensions
alpha = 0.1
dprate = 0.3
dropout = 0.3
K = 5
order = 25
hidden = 16
model = HiGCN(g.num_features, num_classes , hidden, order, alpha , K ,dprate, dropout )

def train(g, model):
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    best_val_acc = 0
    best_test_acc = 0

    features = g.x
    labels = g.y
    train_mask = g.train_mask
    val_mask = g.val_mask
    test_mask = g.test_mask
    for e in range(300):
        logits = model(g)

        pred = logits.argmax(1)
        roc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), F.softmax(logits[test_mask], dim=1).cpu().detach().numpy(), multi_class='ovr')
        
        loss = F.nll_loss(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f},train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f}), roc test : {:.3f}'.format(
                e, loss,train_acc, val_acc, best_val_acc, test_acc, best_test_acc , roc))

train(g, model)
torch.save(model.state_dict(), output_path)
