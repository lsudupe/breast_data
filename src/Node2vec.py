##Node2vec embedding algorithm

# libraries
import numpy as np
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import os
import torch
import pickle
import psutil
from tqdm.notebook import tqdm

# check torch version
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# Load the dictionary
save_path = "/ibex/scratch/medinils/breast_data/data/process/graphs.pkl"
with open(save_path, 'rb') as file:
    graphs = pickle.load(file)
    
# Node2vec
# Convert your networkx graphs to PyTorch Geometric data objects
data_list = [from_networkx(graph) for graph in graphs.values()]

# convert node atributes to tensors
for data in data_list:
    for key, item in data:
        if isinstance(item, np.ndarray):
            data[key] = torch.tensor(item)
            
# save the list
save_path = "/ibex/scratch/medinils/breast_data/data/process/data_list.pt"
torch.save(data_list, save_path)
# read the list
save_path = "/ibex/scratch/medinils/breast_data/data/process/data_list.pt"
data_list = torch.load(save_path)


def memory_usage():
    process = psutil.Process(os.getpid())
    return f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB"

# Dictionary to store embeddings for each graph
graph_embeddings = {}
# Initialize a counter to keep track of processed graphs
processed_graphs = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loop through each graph in the subset and generate embeddings
for patient, data in tqdm(zip(graphs.keys(), data_list), total=len(data_list), desc="Processing Graphs"):
    
    # Initialize Node2Vec model with num_negative_samples parameter
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=15,
                     context_size=5, walks_per_node=150, num_negative_samples=5).to(device)
    loader = model.loader(batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    model.train()
    for epoch in tqdm(range(50), desc="Training Node2Vec", leave=False):
        total_loss = 0  # To compute average loss over all batches
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

    # Get embeddings for all nodes
    model.eval()
    with torch.no_grad():
        embeddings = model() 

    # Aggregate the embeddings using mean (or any other method you choose)
    aggregated_embedding = embeddings.mean(dim=0).cpu().detach().numpy()

    # Store aggregated embeddings in the dictionary
    graph_embeddings[patient] = aggregated_embedding
    
    # Increment the counter and print it
    processed_graphs += 1
    print(f"Processed {processed_graphs} graphs.")
    
    # Print memory usage
    print(memory_usage())
    
# Specify the path for saving
save_path = "/ibex/scratch/medinils/breast_data/data/embeddings/graph_embeddings.npy"

# Save the embeddings to the specified path
np.save(save_path, graph_embeddings)

# Load the embeddings back from the saved file
#graph_embeddings = np.load(save_path, allow_pickle=True)