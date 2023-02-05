import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
from glob import glob
import numpy as np
from torch.utils.data import DataLoader

from da_multitask.data_generator.classification_data_gen import BasicArrayDataset, ResampleArrayDataset
from da_multitask.flow.train_flow import TrainFlow
from da_multitask.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from da_multitask.networks.complete_model import CompleteModel
from da_multitask.networks.backbone_tcn import TCN
from da_multitask.networks.classifier import MultiFCClassifiers


def load_data(folder: str):
    """
    Load all data, 2 classes only (fall and non-fall)
    Args:
        folder:

    Returns:

    """
    train_dict_1 = {0: [], 1: []}
    train_dict_2 = defaultdict(list)
    valid_dict = {0: [], 1: []}

    files = glob(f'{folder}/D*/*.npy')
    print(f'{len(files)} files found')

    for file in files:
        print(f'Reading file: {file}')
        arr = np.load(file)[:, :, 1:]
        valid_idx = np.arange(0, len(arr), 2)
        train_idx = np.setdiff1d(np.arange(len(arr)), valid_idx)

        # if this is D2 dataset, add to
        if file.split(os.sep)[-2] == 'D2':
            d2_class = re.search(r'_task([0-9][0-9]).npy$', file).group(1)
            train_dict_2[d2_class].append(arr)
        # get fall data
        elif file.endswith('_fall.npy'):
            train_dict_1[1].append(arr[train_idx])
            valid_dict[1].append(arr[valid_idx])
        # get all non-fall data
        else:
            train_dict_1[0].append(arr[train_idx])
            valid_dict[0].append(arr[valid_idx])

    train_dict_1 = {key: np.concatenate(value) for key, value in train_dict_1.items()}
    train_dict_2 = {i: np.concatenate(train_dict_2[key]) for i, key in enumerate(sorted(train_dict_2))}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}

    train_set_1 = ResampleArrayDataset(train_dict_1)
    train_set_2 = ResampleArrayDataset(train_dict_2)
    valid_set = BasicArrayDataset(valid_dict)

    return [train_set_1, train_set_2], valid_set


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-de', default='cpu')
    args = parser.parse_args()

    # create data loaders
    train_sets, valid_set = load_data('../../npy_data')
    train_loaders = [DataLoader(train_set, batch_size=8, shuffle=True) for train_set in train_sets]
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    # create model
    backbone = TCN(
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=False
    )
    classifier = MultiFCClassifiers(
        n_features=128,
        n_classes=[train_set.num_classes for train_set in train_sets]
    )
    model = CompleteModel(backbone=backbone, classifier=classifier)

    # create folder to save result
    save_folder = './draft'
    last_exp = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/exp_*')]
    last_exp = max(last_exp) + 1 if len(last_exp) > 0 else 0
    save_folder = f'{save_folder}/exp_{last_exp}'

    # create training config
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    num_epochs = 40

    flow = TrainFlow(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        device=args.device,
        callbacks=[
            ModelCheckpoint(num_epochs, f'{save_folder}/multi_task.pth'),
            # EarlyStop(10)
        ]
    )

    train_log, valid_log = flow.run(
        train_loader=train_loaders,
        valid_loader=[valid_loader, None],  # only valid the first task
        num_epochs=num_epochs
    )

    train_log.to_csv(f'{save_folder}/train.csv', index=False)
    valid_log.to_csv(f'{save_folder}/valid.csv', index=False)

    print("Done!")
