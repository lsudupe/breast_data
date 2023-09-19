# Script for graph creation

## libraries
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay
import networkx as nx
import os
import pickle
from tqdm.notebook import tqdm

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Data load
## select dir
os.getcwd()
data_dir = Path("/ibex/scratch/medinils/breast_data/data/")

##load the data
train_features_dir = data_dir / "train_input" / "moco_features"
test_features_dir = data_dir / "test_input" / "moco_features"
df_train = pd.read_csv(data_dir / "supplementary_data" / "train_metadata.csv")
df_test = pd.read_csv(data_dir / "supplementary_data/test_metadata.csv")

# concatenate y_train and df_train
y_train = pd.read_csv(data_dir / "train_output.csv")
df_train = df_train.merge(y_train, on="Sample ID")

print(f"Training data dimensions: {df_train.shape}")  # (344, 4)
df_train.head()

# Data processing
## create a graph dic to store them
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
            for j in range(i + 1, 3):
                # add the edge between the i-th and j-th point in the triangle
                # the edge weight is the euclidean distance
                p1 = coo[tri[i]]
                p2 = coo[tri[j]]
                distance = np.linalg.norm(p1 - p2)
                G.add_edge(tri[i], tri[j], weight=distance)
                # add features as node attributes
                G.nodes[tri[i]].update({'features': features[tri[i]]})
                G.nodes[tri[j]].update({'features': features[tri[j]]})
                # Add the target label and other info to the graph's attributes
                G.graph['label'] = label
                G.graph['patient_id'] = patient
                G.graph['sample_id'] = sample
                G.graph['center_id'] = center
    # store the graphs for each patient (key)
    graphs[patient] = G

    # Slide-level averaging
    X_train.append(np.mean(features, axis=0))
    y_train.append(label)
    centers_train.append(center)
    patients_train.append(patient)

X_train = np.vstack(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)

# save it
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, "wb") as f:
    pickle.dump(graphs, f)
