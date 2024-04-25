import torch
import torch.nn.functional as F
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv


# Define the class for your model (assuming it's the same as the one used for training)
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

testing_data = torch.load(sys.argv[2], map_location='cpu')  # Assuming testing data is stored in the same format

model_save_path = sys.argv[1]

output_save_path = sys.argv[3]

# Load the model
model = Net(testing_data.x.size(1), 128, 64)
model.load_state_dict(torch.load(model_save_path))

# Define testing function
@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.test_edges).view(-1).sigmoid()
    return out

# Load the testing data

# Perform testing
probabilities = test(testing_data)
# probabilities = [1 if i >= 0.5 else 0 for i in probabilities]
# Convert probabilities tensor to numpy array
probabilities_np = probabilities

# Save probabilities to CSV
df = pd.DataFrame(probabilities_np, columns=['Probability'])
df.to_csv(output_save_path, index=False , header=False )

print("Probabilities saved to probabilities.csv")




