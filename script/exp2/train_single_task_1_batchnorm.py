import os

import torch
import torch.nn as nn
from glob import glob
import numpy as np
from torch.utils.data import DataLoader

from da_multitask.data_generator.augment import Augmenter, ComposeAugmenters, Rotate, TimeWarp
from da_multitask.data_generator.classification_data_gen import BasicArrayDataset, ResampleArrayDataset
from da_multitask.flow.train_flow import TrainFlow
from da_multitask.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from da_multitask.networks.complete_model import CompleteModel
from da_multitask.networks.backbone_tcn import TCN
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

    files = glob(f'{folder}/D*/*.npy')
    print(f'{len(files)} files found')

    for file in files:
        print(f'Reading file: {file}')
        arr = np.load(file)[:, :, 1:]
        valid_idx = np.arange(0, len(arr), 2)
        train_idx = np.setdiff1d(np.arange(len(arr)), valid_idx)

        # if this is D2 dataset, only add to training set
        if file.split(os.sep)[-2] == 'D2':
            train_dict[0].append(arr)
        # get fall data
        elif file.endswith('_fall.npy'):
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

    return train_dict, valid_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-de', default='cpu')
    parser.add_argument('--name', '-n', help='name of the experiment to create a folder to save weights')
    args = parser.parse_args()

    # create model
    backbone = TCN(
        input_shape=(200, 3),
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=False,
        conv_norm='batch',
        attention_conv_norm='batch'
    )
    classifier = FCClassifier(
        n_features=128,
        n_classes=2
    )
    model = CompleteModel(backbone=backbone, classifier=classifier, dropout=0.5)

    # create data loaders
    train_dict, valid_dict = load_data('../../npy_data')

    augmenter = ComposeAugmenters(
        p=1,
        augmenters=[Rotate(p=0.5, angle_range=180, angle_y_range=180, angle_z_range=180),
                    TimeWarp(p=0.5, sigma=0.25, knot_range=4)]
    )

    train_set = ResampleArrayDataset(train_dict, augmenter=augmenter)
    valid_set = BasicArrayDataset(valid_dict)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    # create folder to save result
    save_folder = f'./draft/{args.name}'
    last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
    last_run = max(last_run) + 1 if len(last_run) > 0 else 0
    save_folder = f'{save_folder}/run_{last_run}'

    # create training config
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    num_epochs = 40

    flow = TrainFlow(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        device=args.device,
        callbacks=[
            ModelCheckpoint(num_epochs, f'{save_folder}/single_task.pth'),
            # EarlyStop(10)
        ]
    )

    train_log, valid_log = flow.run(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=num_epochs
    )

    train_log.to_csv(f'{save_folder}/train.csv', index=False)
    valid_log.to_csv(f'{save_folder}/valid.csv', index=False)

    print("Done!")