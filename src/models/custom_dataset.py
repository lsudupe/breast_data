# create a custom dataset class
import torch
from torch_geometric.data import Dataset, Data

class BreastData(Dataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        super(BreastData, self).__init__(root, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def processed_file_names(self):
        # This function expects names of processed files, but we can return a dummy list
        return ['data.pt']

    def process(self):
        # save the list
        torch.save(self.data_list, self.procedded_dir + '/data.pt')

    def load_processed(self):
        # load the data from the processed file
        self.data_list = torch.load(self.processed_dir + '/data.pt')