# Plot GNN models metrics

## libraries
import os
import torch
import matplotlib.pyplot as plt

model_type_list = ['gcn', 'gat', 'graphsage']

# Placeholder for metrics
all_metrics = {}

# Load metrics
for model_type in model_type_list:
    metrics_dir = f"/ibex/scratch/medinils/breast_data/results/{model_type.upper()}/metrics"
    all_metrics[model_type] = torch.load(os.path.join(metrics_dir, 'metrics.pt'))

plots_dir = f"/ibex/scratch/medinils/breast_data/results/plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot accuracy for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['accuracy_test'], label=model_type.upper())
    plt.plot(all_metrics[model_type]['accuracy_train'], '--', label=f"Train {model_type.upper()}")

plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "accuracy_over_epochs.pdf"))
plt.show()

# Plot F1 for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['f1'], label=model_type.upper())

plt.title('Test f1 Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('f1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "f1_over_epochs.pdf"))
plt.show()

# Plot precision for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['precision'], label=model_type.upper())

plt.title('Test precision Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('precision')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "precision_over_epochs.pdf"))
plt.show()

# Plot recall for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['recall'], label=model_type.upper())

plt.title('Test recall Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('recall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "recall_over_epochs.pdf"))
plt.show()

# Plot loss convergence for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['train_loss'], label=model_type.upper())

plt.title('Test loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "train_loss_over_epochs.pdf"))
plt.show()

# Plot time per epoch for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['time_per_epoch'], label=model_type.upper())

plt.title('Test time_per_epoch Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('time_per_epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "time_per_epoch_over_epochs.pdf"))
plt.show()

# Plot memory usage per epoch for all models
plt.figure(figsize=(10, 6))
for model_type in model_type_list:
    plt.plot(all_metrics[model_type]['memory_usage'], label=model_type.upper())

plt.title('Test memory_usage Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('memory_usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(plots_dir, "memory_usage_over_epochs.pdf"))
plt.show()

