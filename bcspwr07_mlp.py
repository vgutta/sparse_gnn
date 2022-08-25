import torch
import torch.nn.functional as F
from torch.nn import Linear
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomNodeSplit
import scipy as sp
import scipy.io
import numpy as np

transform = RandomNodeSplit('train_rest', key='y')

bcspwr = sp.io.mmread("./matrix_market_graphs/bcspwr07.mtx")
mmg1 = nx.from_scipy_sparse_matrix(bcspwr)

values = bcspwr.data
indices = np.vstack((bcspwr.row, bcspwr.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = bcspwr.shape

sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

data = from_networkx(mmg1)
csr = sparse_tensor.to_sparse_csr()
data.x = csr
data.y = torch.randint(0,10, [data.num_nodes,])  # random y values
data = transform(data)

data.num_features = 1612
data.num_classes = 10

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(data.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, data.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      #loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
