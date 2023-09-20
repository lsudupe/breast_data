# Run GNN models

## libraries
import sys
sys.path.append('/ibex/scratch/medinils/breast_data/')
import torch
import os
from collections import defaultdict
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from src.models.custom_dataset import BreastData
from src.models.GNN import GNN

## read dataset
datasetLaura = BreastData(data_list, root="/ibex/scratch/medinils/breast_data/data/process/")

## extract patient, sample and center id
patient_ids = [data.patient_id for data in datasetLaura]
sample_ids = [data.sample_id for data in datasetLaura]
center_ids = [data.center_id for data in datasetLaura]

## Create a dictionary where keys are patient_ids and values are list of indices associated with that patient_id
patient_to_indices = defaultdict(list)
for idx, data in enumerate(datasetLaura):
    patient_id = data.patient_id  # assuming `patient_id` attribute exists in your data object
    patient_to_indices[patient_id].append(idx)

## Convert the dictionary values (lists of indices) to a list
grouped_indices = list(patient_to_indices.values())

## split data into training and test sets
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_grouped_idx, test_grouped_idx in kf.split(grouped_indices):
    # Flatten the grouped indices to get actual dataset indices
    train_idx = [idx for group in train_grouped_idx for idx in grouped_indices[group]]
    test_idx = [idx for group in test_grouped_idx for idx in grouped_indices[group]]
    train_dataset = [datasetLaura[i] for i in train_idx]
    test_dataset = [datasetLaura[i] for i in test_idx]

## print the number of graphs in train and test sets
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

## Batching of graphs
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

## print the batches
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

## train function
def train(model, optimizer, criterion, loader):
    model.train()

    total_loss = 0  # for optionally tracking the average training loss
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)["output"]  # Note: using ["output"]
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()  # optionally tracking the average training loss

    return total_loss / len(loader)  # optionally return the average training loss

## test function
def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)["output"]
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)

## lets evaluate the three models
## list of models
model_type = ['gcn', 'gat', 'graphsage']

for model_type in model_type_list:
    # Initialize model, optimizer, criterion
    model = GNN(dataset=datasetLaura, hidden_channels=64, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, 50):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f'Model: {model_type.upper()}, Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Save the embeddings
    embeddings_dir = f"/ibex/scratch/medinils/breast_data/results/{model_type.upper()}/embeddings"
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    torch.save(model.embeddings, os.path.join(embeddings_dir, 'embeddings.pt'))

    # Plotting training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 50), train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, 50), test_accuracies, label='Test Accuracy', color='red', marker='o')
    plt.title(f'{model_type.upper()} Training and Testing Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plots_dir = f"/ibex/scratch/medinils/breast_data/results/{model_type.upper()}/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, "accuracy_over_epochs.pdf"))
    plt.show()