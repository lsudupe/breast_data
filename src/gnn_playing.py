import sys
sys.path.append('/ibex/scratch/medinils/breast_data/')
from pathlib import Path
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Dataset
import pickle
import networkx as nx


# Load the dictionary
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, 'rb') as file:
    graphs = pickle.load(file)

data_list = []
for graph in graphs.values():
    data = from_networkx(graph, group_node_attrs=["features"])
    data.y = torch.tensor([graph.graph['label']], dtype=torch.long)
    data_list.append(data)


from src.models.custom_dataset import BreastData

datasetLaura = BreastData(data_list=data_list, root="/ibex/scratch/medinils/breast_data/data/process/")

print(f'Dataset: {datasetLaura}:')
print(f'Number of features: {datasetLaura.num_features}')
print(f'Number of classes: {datasetLaura.num_classes}')


# check first graph
data = datasetLaura[0]

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

G1 = next(iter(graphs.values()))

patient_ids = [data.patient_id for data in datasetLaura]
sample_ids = [data.sample_id for data in datasetLaura]
center_ids = [data.center_id for data in datasetLaura]

from collections import defaultdict
# Create a dictionary where keys are patient_ids and values are list of indices associated with that patient_id
patient_to_indices = defaultdict(list)
for idx, data in enumerate(datasetLaura):
    patient_id = data.patient_id  # assuming `patient_id` attribute exists in your data object
    patient_to_indices[patient_id].append(idx)

# Convert the dictionary values (lists of indices) to a list
grouped_indices = list(patient_to_indices.values())

from sklearn.model_selection import KFold

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_grouped_idx, test_grouped_idx in kf.split(grouped_indices):
    # Flatten the grouped indices to get actual dataset indices
    train_idx = [idx for group in train_grouped_idx for idx in grouped_indices[group]]
    test_idx = [idx for group in test_grouped_idx for idx in grouped_indices[group]]

    train_dataset = [datasetLaura[i] for i in train_idx]
    test_dataset = [datasetLaura[i] for i in test_idx]


print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

## Batching of graphs

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

## Training GNN
from src.models.GNN import GCN

model = GNN(dataset=datasetLaura, hidden_channels=64)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 50):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


## add epoch plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
plt.plot(epochs, test_acc, label='Test Accuracy', color='red', marker='o')
plt.title('Training and Testing Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## save the plot
# Specify the directory and filename
save_dir = "/ibex/scratch/medinils/breast_data/results/GNN"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "accuracy_over_epochs.pdf")

# Save the figure as a PDF
plt.savefig(save_path)

