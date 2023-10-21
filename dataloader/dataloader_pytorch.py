import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset_NoLabels(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset_NoLabels, self).__init__()

        X_train = dataset["samples"]

        # make sure the Channels in second dim
        if X_train.shape[1:].index(min(X_train.shape[1:])) != 0:
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # if len(X_train.shape) < 3:
        #     X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        # if X_train.shape.index(min(X_train.shape)) != 1:
        #     X_train = X_train.permute(0, 2, 1)
        if X_train.shape[1:].index(min(X_train.shape[1:])) != 0:
            X_train = X_train.permute(0, 2, 1)


        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

def data_generator(data_path, labels=True):

    test_dataset = torch.load(os.path.join(data_path))
    # "E:/test_x_y.pt"
    # "E:\MyCode\TS-TCC\data\sleepEDF\train.pt"
    if labels == True:
        test_dataset = Load_Dataset(test_dataset)
    else:
        test_dataset = Load_Dataset_NoLabels(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return test_loader
