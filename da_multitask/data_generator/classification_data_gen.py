import numpy as np
import torch as tr
from torch.utils.data import Dataset


class BasicArrayDataset(Dataset):
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
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self) -> int:
        return len(self.label)


class ResampleArrayDataset(Dataset):
    def __init__(self, label_data_dict: dict, shuffle: bool = True):
        """

        Args:
            label_data_dict: key: label (int), value: data [n, ..., channel]
            shuffle: shuffle data after each epoch

        """
        self.shuffle = shuffle
        # key: label (int); value: data [n, ..., channel]
        self.label_data_dict = label_data_dict
        # key: label; value: index of the last called instance
        self.label_pick_idx = {}

        print('Label distribution:')
        for cls, arr in self.label_data_dict.items():
            print(cls, ':', len(arr))
            self.label_data_dict[cls] = tr.from_numpy(arr).float()
            self.label_pick_idx[cls] = 0

        # calculate dataset size
        self.dataset_len = sum(len(arr) for arr in self.label_data_dict.values())
        self.mean_class_len = self.dataset_len / len(self.label_data_dict)

    def __getitem__(self, index):
        label = int(index // self.mean_class_len)
        data = self.label_data_dict[label][self.label_pick_idx[label]]

        # update index
        self.label_pick_idx[label] += 1
        # if reach epoch end of this class
        if self.label_pick_idx[label] == len(self.label_data_dict[label]):
            self.label_pick_idx[label] = 0
            self._shuffle_class_index(label)

        return data, label

    def _shuffle_class_index(self, cls: int):
        if self.shuffle:
            self.label_data_dict[cls] = self.label_data_dict[cls][
                tr.randperm(len(self.label_data_dict[cls]))
            ]

    def __len__(self) -> int:
        return self.dataset_len


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = ResampleArrayDataset({
        0: np.arange(1) + 0,
        1: np.arange(2) + 100,
        2: np.arange(5) + 200,
        3: np.arange(10) + 300,
    }, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(10):
        print(f'\nepoch {epoch}')
        iter_loader = iter(dataloader)
        len_loader = len(dataloader) - 1
        for batch in range(len_loader):
            _data, _label = next(iter_loader)
            print('batch: {}; data: {}; label: {}'.format(batch, _data, _label))