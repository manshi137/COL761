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
import pandas as pd
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




def test(g, model,output_file):
    test_mask = torch.full((g.num_nodes,), True, dtype=torch.bool)

    for i in range (g.num_nodes):
        if g.y[i]== -1: 
            test_mask[i] = False
    features = g.x
    labels = g.y
    # print(g.HL)
    logits = model(g)
    probabilities = torch.exp(logits)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    roc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), F.softmax(logits[test_mask], dim=1).cpu().detach().numpy(), multi_class='ovo',average='macro')

    print('Test accuracy:', test_acc.item())
    print('ROC AUC score:', roc)
    probs_numpy = probabilities.cpu().detach().numpy()
    pd.DataFrame(probs_numpy).to_csv(output_file, header=False, index=False)
    
    # Create DataFrame with probabilities
    # df_probs = pd.DataFrame(probs_numpy, columns=[f'Class_{i}' for i in range(probs_numpy.shape[1])])
    
    # Save DataFrame to CSV
if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    g = torch.load(dataset_path ,map_location='cpu')
    num_classes = 5
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes))
    G.add_edges_from(g.edge_index.numpy().transpose())
    g.HL = creat_L_SparseTensor(G, maxCliqueSize=25 )
    # g = torch.load(dataset_path, map_location='cpu')  # Load your graph data
    alpha = 0.1
    dprate = 0.3
    dropout = 0.3
    K = 5
    order = 25
    hidden = 16
    model = HiGCN(g.num_features, num_classes , hidden, order, alpha , K ,dprate, dropout )
    model.load_state_dict(torch.load(model_path))
    test(g, model,output_path)