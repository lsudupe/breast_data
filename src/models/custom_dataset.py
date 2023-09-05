# create a custom dataset class

class BreastData(Dataset):
    def __int__(self, data_list, root, transform=None, pre_transform=None):
        super(BreastData, self).__init__(root, transform, pre_transform)
        self._data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
