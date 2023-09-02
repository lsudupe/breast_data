from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from torch_geometric.nn import Node2Vec
import os
import psutil
from tqdm.notebook import tqdm
from ipywidgets import FloatProgress
from torch_geometric.utils import from_networkx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("a")
DATA_DIR = Path("/ibex/scratch/medinils/breast_data/data/")
OUTPUT_DIR = Path("results/example-training-job/")
OUTPUT_FILENAME = OUTPUT_DIR / "model.joblib"
SEED = 42
TEST_SIZE = 0.1
TRAIN_N_JOBS = -1
VERBOSITY = 10

print("b")
##load the data
train_features_dir = data_dir / "train_input" / "moco_features"
test_features_dir = data_dir / "test_input" / "moco_features"
df_train = pd.read_csv(data_dir/"supplementary_data"/"train_metadata.csv")
df_test = pd.read_csv(data_dir/"supplementary_data/test_metadata.csv")

# concatenate y_train and df_train
y_train = pd.read_csv(data_dir  / "train_output.csv")
df_train = df_train.merge(y_train, on="Sample ID")

#create a graph dic to store them
graphs = {}

X_train = []
y_train = []
centers_train = []
patients_train = []

for sample, patient, center, label in tqdm(
    df_train[["Sample ID", "Patient ID", "Center ID", "Target"]].values
):
    # coordinates and features each iteration one sample
    _features = np.load(train_features_dir / sample)
    # coo and features, separate them
    coo, features = _features[:, 1:3], _features[:, 3:]
    # delaunay triangulation
    tri = Delaunay(coo)
    # Create the graph to add the
    G = nx.Graph()

    # add the edges for each Delaunay triangulation
    for tri in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                # add the edge between the i-th and j-th point in the triangle
                # the edge weight is the euclidean distance
                p1 = coo[tri[i]]
                p2 = coo[tri[j]]
                distance = np.linalg.norm(p1 - p2)
                G.add_edge(tri[i], tri[j], weight =distance)
                # add features as node attributes
                G.nodes[tri[i]].update({'features': features[tri[i]]})
                G.nodes[tri[j]].update({'features': features[tri[j]]})
    # store the graphs for each patient (key)
    graphs[patient] = G
    
    # Slide-level averaging
    X_train.append(np.mean(features, axis=0))
    y_train.append(label)
    centers_train.append(center)
    patients_train.append(patient)

import psutil
print("c")
def memory_usage():
    process = psutil.Process(os.getpid())
    return f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB"

from tqdm.notebook import tqdm
from ipywidgets import FloatProgress
from torch_geometric.utils import from_networkx

# Dictionary to store embeddings for each graph
graph_embeddings = {}

# Convert your networkx graphs to PyTorch Geometric data objects
data_list = [from_networkx(graph) for graph in graphs.values()]
print("d")

# Loop through each graph and generate embeddings
for patient, data in tqdm(zip(graphs.keys(), data_list), desc="Generating embeddings"):
    # Initialize Node2Vec model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=15,
                     context_size=10, walks_per_node=50).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        loss = model.loss(data.edge_index)
        loss.backward()
        optimizer.step()

    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model()

    # Aggregate the embeddings using mean (or any other method you choose)
    aggregated_embedding = embeddings.mean(dim=0).cpu().numpy()

    # Store aggregated embeddings in the dictionary
    graph_embeddings[patient] = aggregated_embedding
    
print("d")
# Save the embeddings to a file in the specified directory
np.save(data_dir / 'embeddings/graph_embeddings.npy', graph_embeddings)

