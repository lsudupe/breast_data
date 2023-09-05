import sys
sys.path.append('/ibex/scratch/medinils/breast_data/')
from pathlib import Path
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Dataset




from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
data = dataset[0]  # Get the first graph object.
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]
train_dataset
test_dataset

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# Load the dictionary
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, 'rb') as file:
    graphs = pickle.load(file)

# Node2vec
# Convert your networkx graphs to PyTorch Geometric data objects
data_list = [from_networkx(graph) for graph in graphs.values()]

from src.models.custom_dataset import BreastData
from src.models.custom_dataset import BreastData


