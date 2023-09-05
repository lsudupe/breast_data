

from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import os
import pickle
import psutil
from tqdm.notebook import tqdm




# Load the dictionary
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, 'rb') as file:
    graphs = pickle.load(file)



key_list = list(graphs.keys())
G1 = graphs[key_list[0]]
g = nx.Graph()

print(f'number of nodes: {G1.number_of_nodes()}')
print(f'node features: {nx.get_node_attributes(G1, "features")}')
print(f'number of edges: {G1.number_of_edges()}')
print(f'node features: {nx.get_edge_attributes(G1)}')

print(f'number of isolated nodes: {nx.number_of_isolates(G1)}')
print(f'number of self-loops: {nx.number_of_selfloops(G1)}')
print(f"is the graph directed: {nx.is_directed(G1)}")

degrees = [val for (node, val) in G1.degree()]
print(len(degrees))
print(sum(degrees))
print(f'average degree: {nx.average_degree_connectivity(G1)}')
print(f'average connectivity degree in the neighborhood: {nx.average_neighbor_degree(G1)}')

plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=50)
plt.xlabel("node degree")
plt.show()

## lets check where the top 10 nodes with the highest degree are
pos = nx.spring_layout(G1, seed=42)
cent = nx.degree_centrality(G1)
node_size = list(map(lambda x: x * 500, cent.values()))
cent_array = np.array(list(cent.values()))
threshold = sorted(cent_array, reverse=True)[10]
print("threshold", threshold)
cent_bin = np.where(cent_array >= threshold, 1, 0.1)
# Determine node colors: red for nodes with centrality above the threshold, otherwise use another color (e.g., blue)
node_colors = ['red' if is_big else 'blue' for is_big in cent_bin == 1]
# Increase the size for nodes with centrality above the threshold
scale_factor = 2  # Adjust this value based on how much bigger you want those nodes to be
node_size = [size * scale_factor if is_big else size for size, is_big in zip(node_size, cent_bin == 1)]# plot
plt.figure(figsize=(12, 12))
nodes = nx.draw_networkx_nodes(G1, pos, node_size=node_size,
                               node_color=node_colors,
                               nodelist=list(cent.keys()),
                               alpha=cent_bin)
edges = nx.draw_networkx_edges(G1, pos, width=0.25, alpha=0.3)
plt.show()

#plot
pos = nx.spring_layout(G1, k=0.10)
nx.draw_networkx(G1, pos, node_size=10, width= 0.3,with_labels=False)
plt.show()