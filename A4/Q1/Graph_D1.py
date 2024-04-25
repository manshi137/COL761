import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool
import random 
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import FastRGCNConv, RGCNConv , GATConv ,GMMConv
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
from torch.nn import Linear, ReLU, Sequential


from sklearn.metrics import roc_auc_score


import sys
Conv = RGCNConv

def convert_x_to_float(data):
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    return data

def collate(data_list):
    return Batch.from_data_list(data_list)

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, dataset[0].x.size(1) * 32),
        )
        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 32 * 64),
        )
        self.conv1 = GMMConv(dataset[0].x.size(1), 32 ,aggr='mean' , dim=3 , kernel_size=5)
        self.conv2 = GMMConv(32, 64, aggr='mean', dim=3 , kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128) 
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)
    
def training_phase(dataset, model_path):
    
    Conv = RGCNConv

    torch.manual_seed(12345)
    import time
    def convert_x_to_float(data):
        data.x = data.x.float()
        # print(dense_to_sparse(data.edge_index))
        # data.edge_index = dense_to_sparse( data.edge_index)[e]
        
        data.edge_attr = data.edge_attr.float()
        return data

    # dataset = torch.load("GC_D1.pt" , map_location='cpu')
    # print(dataset)
    print(type(dataset))
    print(len(dataset))
    # print(dataset)
    print(dataset[0])

    # Convert x to float before splitting the dataset
    numz = 0 
    num1 = 0 
    oth = 0 
    for i in range(len(dataset)):
        # print(dataset[i].y)
        if(dataset[i].y == 0 ):
            # dataset[i].y = 1
            # dataset[i].y = 1
            numz+=1   
        elif(dataset[i].y == 1 ):
            num1+=1   
            # dataset[i].y = 0

        else :
            oth+=1 
        dataset[i] = convert_x_to_float(dataset[i])

    print(dataset[0])

    print(f'num1 = {num1}, numz = {numz}, oth = {oth}')
    random.shuffle(dataset)

    num_graphs = len(dataset)

    minority_label = 1

    # Find the indices of graphs belonging to the minority class
    minority_indices = [i for i, data in enumerate(dataset) if data.y == minority_label]

    # Determine the number of graphs needed to balance the dataset
    num_majority = len(dataset) - len(minority_indices)
    print(num_majority , len(minority_indices))
    # Duplicate graphs from the minority class to balance the dataset
    balanced_dataset = []
    for i in range(num_majority):
        # Select a random graph from the minority class
        idx = random.choice(minority_indices)
        balanced_dataset.append(dataset[idx].clone())
    # Concatenate the original majority class graphs with the duplicated minority class graphs
    balanced_dataset.extend([data for i, data in enumerate(dataset) if i not in minority_indices])

    # Shuffle the balanced dataset
    random.shuffle(balanced_dataset)

    num_graphs = len(balanced_dataset)
    print("numgraphs =" , num_graphs)
    # Use the balanced dataset for training and testing
    train_dataset = balanced_dataset[:(num_graphs*8)//10]

    val_dataset = balanced_dataset[(num_graphs*8)//10 : (num_graphs*9)//10]
    test_dataset = balanced_dataset[(num_graphs*9)//10:]
    # print("pos = " , dataset[0].pos)
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    def collate(data_list):
        # 'data_list' is a list of your custom Data objects
        batch = Batch.from_data_list(data_list)
        return batch
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=collate)



    # transform = T.Cartesian(cat=False)
    hidden_size = 16
    dropout = 0.3
    ff_hidden_size = 256
    num_hidden_layers = 1
    learning_rate = 0.001
    print(dataset[0].edge_attr.size())
    def normalized_cut_2d(edge_index,edge_attr, s):
        # print('row = ' , row ,'col = ', col )
        edge_attr_transposed = edge_attr.t()[0]
        # edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
        return normalized_cut(edge_index, edge_attr_transposed, num_nodes=s)


    # Define GCN model
    class GCN(torch.nn.Module):
        def __init__(self, dataset, hidden_channels):
            super(GCN, self).__init__()
            nn1 = Sequential(
                Linear(2, 25),
                ReLU(),
                Linear(25, dataset[0].x.size(1) * 32),
            )
            nn2 = Sequential(
                Linear(2, 25),
                ReLU(),
                Linear(25, 32 * 64),
            )
            self.conv1 = GMMConv(dataset[0].x.size(1), 32 ,aggr='mean' , dim=3 , kernel_size=5)
            self.conv2 = GMMConv(32, 64, aggr='mean', dim=3 , kernel_size=5)
            self.fc1 = torch.nn.Linear(64, 128) 
            self.fc2 = torch.nn.Linear(128, 2)

        def forward(self, data):
            x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            # weight = normalized_cut_2d(data.edge_index, data.edge_attr ,data.x.size(0))
            # cluster = graclus(data.edge_index, weight, data.x.size(0))
            # data.edge_attr = None
            # data = max_pool(cluster, data)

            x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
            # weight = normalized_cut_2d(data.edge_index, data.edge_attr, data.x.size(0))
            # cluster = graclus(data.edge_index, weight, data.x.size(0))
            # x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_mean_pool(x, data.batch)
            x = F.elu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            return F.log_softmax(self.fc2(x), dim=1)



    model = GCN(train_dataset, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        i = 0 
        l = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            l += loss 
            i += 1 
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return l / i

    def test(loader):
        model.eval()
        y_true = []
        y_pred = [] 
        with torch.no_grad():
            correct = 0
            r = 0 
            for data in loader:  # Iterate in batches over the training/test dataset.
                out = model(data)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                probs = torch.exp(out)  # Applying exponential function to logits
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
                y_true.extend(data.y.tolist())
                y_pred.extend(probs[:, 1].tolist())  # Probabilities for class 1

            return correct / len(loader.dataset) , roc_auc_score(y_true , y_pred)  # Derive ratio of correct predictions.
    test_acc = test(test_loader)
    print(f'acc before training : {test_acc}')
    saved_flags = {90: False, 91: False, 92: False, 93: False, 94: False,
                95: False, 96: False, 97: False, 98: False, 99: False}

    best_acc = 0 
    for epoch in range(1, 171):
        loss =  train()
        
        train_acc, train_roc = test(train_loader)
        test_acc, test_roc = test(test_loader)

        if test_acc > 0.90 and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with test accuracy above 90% ')        
        print(f'Epoch: {epoch:03d}, Loss : {loss} Train Acc: {train_acc:.4f}, Train roc :{train_roc:.4f}, Test Acc: {test_acc:.4f} , Test roc :{test_roc:.4f}')


        # torch.save(model.state_dict(), model_save_path)
        # print(f"Model saved to {model_save_path}")

if len(sys.argv) != 4 and len(sys.argv) !=5:
    print("Usage: python script.py train dataset_path model_save_path or python script.py test model_path dataset_path output_path")
    sys.exit(1)
    
if sys.argv[1] == "train":
    if len(sys.argv) != 4:
        print("Usage: python script.py train train_dataset model_path")
        sys.exit(1)
    
    dataset_path = sys.argv[2]
    model_save_path = sys.argv[3]

    dataset = torch.load(dataset_path)
    training_phase(dataset, model_save_path)

elif sys.argv[1] == "test":
    if len(sys.argv) != 5:
        print("Usage: python script.py test model_path test_dataset output_lable_path")
        sys.exit(1)

    model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]

    dataset = torch.load(dataset_path)
    for i in range(len(dataset)):
        dataset[i] = convert_x_to_float(dataset[i])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    model = GCN(dataset,hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    probabilities = []
    model.eval()
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            probs = F.softmax(out, dim=1)[:, 1]  # Probabilities for class 1
            probabilities.extend(probs.cpu().numpy())

    test_y_true = []
    test_y_pred_prob = []
    for data in test_loader:
        test_y_true.extend(data.y.tolist())
        out = model(data)
        test_y_pred_prob.extend(F.softmax(out, dim=1)[:, 1].tolist())
    test_roc_auc = roc_auc_score(test_y_true, test_y_pred_prob)
    print(f'Test ROC AUC: {test_roc_auc:.4f}')

    with open(output_path, 'w') as f:
        for prob in probabilities[:-1]:
            f.write(f"{prob}\n")
        f.write(f"{probabilities[-1]}")

    print(f"Probabilities saved to {output_path}")
else:
    print("Invalid arguments")
