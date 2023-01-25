import numpy as np
import torch as tr
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from da_multitask.constants import DEVICE


class BasicDataset(Dataset):
    def __init__(self, label_data_dict: dict):
        """

        Args:
            label_data_dict: key: label (int), value: data [n, ..., channel]
        """
        print('Label distribution:')
        for k, v in label_data_dict.items():
            print(k, ':', len(v))

        self.data = []
        self.label = []
        for label, data in label_data_dict.items():
            self.data.append(data)
            self.label.append([label] * len(data))

        self.data = np.concatenate(self.data)
        self.label = np.concatenate(self.label)

        self.data = tr.from_numpy(self.data).float()
        self.label = tr.from_numpy(self.label).long()

    def __getitem__(self, index):
        data = self.data[index].to(DEVICE)
        label = self.label[index].to(DEVICE)

        return data, label

    def __len__(self) -> int:
        return self.label.shape[0]


class ResampleDataset(Dataset):
    def __init__(self, label_data_dict: dict):
        """

        Args:
            label_data_dict: key: label (int), value: data [n, ..., channel]
        """
        print('Label distribution:')
        for k, v in label_data_dict.items():
            print(k, ':', len(v))
            label_data_dict[k] = tr.from_numpy(v).float()

    def __getitem__(self, index):
        data = self.data[index].to(DEVICE)
        label = self.label[index].to(DEVICE)

        return data, label

    def __len__(self) -> int:
        return self.label.shape[0]
