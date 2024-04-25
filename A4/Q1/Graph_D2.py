import torch
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv, global_max_pool, global_add_pool
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import sys

# Load model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(579)
        self.conv1 = GATv2Conv(3, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_nodes = x.size(0)
        degrees = torch.zeros(num_nodes, 2, dtype=torch.float)
        src, dst = edge_index
        degrees[:, 0] = torch.bincount(dst, minlength=num_nodes)
        degrees[:, 1] = torch.bincount(src, minlength=num_nodes)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = torch.cat([x, degrees], dim=1)

        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)
        # x = F.dropout(x, p=0.3, training=self.training)
        # x = F.tanh(x)
        x = self.lin2(x)
        return x
        
# Load dataset
if len(sys.argv) != 4 and len(sys.argv) !=5:
    print("Usage: python script.py train dataset_path model_save_path or python script.py test model_path dataset_path output_path")
    sys.exit(1)


if sys.argv[1] == "train":
    dataset_path = sys.argv[2]
    model_save_path = sys.argv[3]

    # Load dataset
    dataset = torch.load(dataset_path)

    train_percentage = 0.9
    train_size = int(train_percentage * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.08)
    criterion = torch.nn.functional.cross_entropy

    scheduler = StepLR(optimizer, step_size=5, gamma=0.08)

    # Training function
    def train():
        model.train()
        total_loss = 0

        for data in train_loader:
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    def validate(loader):
        model.eval()
        y_true = []
        y_pred_prob = []

        for data in loader:
            out = model(data)
            y_true.extend(data.y.tolist())
            y_pred_prob.extend(F.softmax(out, dim =1)[:, 1].tolist())  # Probabilities for class 1

        return roc_auc_score(y_true, y_pred_prob)

    # Training loop
    best_acc = 0 
    done =0
    for epoch in range(1, 30):
        train_loss = train()
        val_auc = validate(test_loader)
        if val_auc > best_acc and val_auc > 0.6:
            best_acc = val_auc
            done = 1
            torch.save(model.state_dict(), model_save_path)
            # print(f'Model saved with test accuracy above 90% ') 
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val ROC AUC: {val_auc:.4f}')
        scheduler.step()


    test_auc = validate(test_loader)
    print(f'Test ROC AUC: {test_auc:.4f}')

    # Save trained model
    if done == 1:
        torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")

elif sys.argv[1] == "test":
    model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]

    model = GCN(hidden_channels=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Load dataset
    dataset = torch.load(dataset_path)
    test_loader = DataLoader(dataset, shuffle=False)

    # Test phase
    model.eval()
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            probs = F.softmax(out, dim=1)[:, 1]  # Probabilities for class 1
            probabilities.extend(probs.cpu().numpy())

    # Calculate ROC-AUC for test dataset
    test_y_true = []
    test_y_pred_prob = []
    for data in test_loader:
        test_y_true.extend(data.y.tolist())
        out = model(data)
        test_y_pred_prob.extend(F.softmax(out, dim=1)[:, 1].tolist())
    test_roc_auc = roc_auc_score(test_y_true, test_y_pred_prob)
    print(f'Test ROC AUC: {test_roc_auc:.4f}')

    # Save probabilities to file
    with open(output_path, 'w') as f:
        for prob in probabilities[:-1]:
            f.write(f"{prob}\n")
        f.write(f"{probabilities[-1]}")

    print(f"Probabilities saved to {output_path}")
else:
    print("Invalid arguments")