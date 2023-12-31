# Read the graphs and create a pyorch dataset
## libraries
from torch_geometric.utils import from_networkx
import pickle
import torch
from models.custom_dataset import BreastData

## Load the dictionary
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, 'rb') as file:
    graphs = pickle.load(file)

## from dic to list
data_list = []
for graph in graphs.values():
    data = from_networkx(graph, group_node_attrs=["features"])
    data.y = torch.tensor([graph.graph['label']], dtype=torch.long)
    data_list.append(data)

## create dataset
datasetLaura = BreastData(data_list=data_list, root="/ibex/scratch/medinils/breast_data/data/process/")

print(f'Dataset: {datasetLaura}:')
print(f'Number of features: {datasetLaura.num_features}')
print(f'Number of classes: {datasetLaura.num_classes}')

## check first graph
data = datasetLaura[0]

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')




