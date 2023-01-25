import torch
import torch.nn as nn
from glob import glob
import numpy as np
from torch.utils.data import DataLoader

from da_multitask.data_generator.data_generator import BasicDataset
from da_multitask.flow.model_loop import train_loop, test_loop
from da_multitask.networks.complete_model import CompleteModel
from da_multitask.networks.backbone import TCN
from da_multitask.networks.classifier import FCClassifier


def load_data(folder: str):
    """
    Load all data, 2 classes only (fall and non-fall)
    Args:
        folder:

    Returns:

    """
    train_dict = {0: [], 1: []}
    valid_dict = {0: [], 1: []}

    files = glob(f'{folder}/*.npy')

    for file in files:
        arr = np.load(file)[:, :, 1:]
        valid_idx = np.arange(0, len(arr), 3)
        train_idx = np.setdiff1d(np.arange(len(arr)), valid_idx)

        # get fall data
        if file.endswith('_fall.npy'):
            train_dict[1].append(arr[train_idx])
            valid_dict[1].append(arr[valid_idx])
        # get all non-fall data
        else:
            train_dict[0].append(arr[train_idx])
            valid_dict[0].append(arr[valid_idx])

    assert train_dict.keys() == valid_dict.keys(), 'train and valid mismatch'

    for key in train_dict.keys():
        train_dict[key] = np.concatenate(train_dict[key])
        valid_dict[key] = np.concatenate(valid_dict[key])

    train_set = BasicDataset(train_dict)
    valid_set = BasicDataset(valid_dict)

    return train_set, valid_set


if __name__ == '__main__':
    # create model
    backbone = TCN(
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=False
    )
    classifier = FCClassifier(
        n_features=128,
        n_classes=2
    )
    model = CompleteModel(backbone=backbone, classifier=classifier)

    # create data loaders
    train_set, valid_set = load_data('../da_multitask/public_datasets/draft')
    train_loader = DataLoader(train_set, batch_size=8)
    valid_loader = DataLoader(valid_set, batch_size=8)

    # create training config
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(valid_loader, model, loss_fn)
    print("Done!")
