import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
from glob import glob
import numpy as np
from torch.utils.data import DataLoader

from da_multitask.data_generator.augment import ComposeAugmenters, Rotate, TimeWarp
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
    train_dict_1 = {0: [], 1: []}  # D1, 2 D1 classes
    train_dict_2 = defaultdict(list)  # D1 fall + D2, 1 D1 fall class + 21 D2 classes
    valid_dict = {0: [], 1: []}

    # GET D2
    files = sorted(glob(f'{folder}/D2/*/*.npy'))
    print(f'{len(files)} files found for D2')
    for file in files:
        # print(f'Reading file: {file}')
        arr = np.load(file)[:, :, 1:]

        d2_class = file.split('/')[-2]
        train_dict_2[f'D2_{d2_class}'].append(arr)

    # GET D1, both train and valid
    files = sorted(glob(f'{folder}/D1/*/*.npy'))
    print(f'{len(files)} files found for D1')
    for i, file in enumerate(files):
        # print(f'Reading file: {file}')
        arr = np.load(file)[:, :, 1:]

        # use this file (sequence) for train or valid
        is_train = (i % 2 == 0)
        # get label: fall: 1; non-fall: 0
        file_label = int(os.path.split(file)[0].endswith('_fall'))
        
        if is_train:
            train_dict_1[file_label].append(arr)
            # only add to train_dict_2 if this label is fall
            if file_label == 1:
                train_dict_2[file_label].append(arr)
        # if is valid
        else:
            valid_dict[file_label].append(arr)

    train_dict_1 = {key: np.concatenate(value) for key, value in train_dict_1.items()}
    train_dict_2 = {i: np.concatenate(val) for i, val in enumerate(train_dict_2.values())}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}

    return [train_dict_1, train_dict_2], valid_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, required=True)
    parser.add_argument('--name', '-n', default='exp4',
                        help='name of the experiment to create a folder to save weights')
    parser.add_argument('--data-folder', '-data', default='/home/ducanh/projects/npy_data_seq/',
                        help='path to data folder')
    parser.add_argument('--output-folder', '-o', default='./draft',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    # create data loaders
    train_dicts, valid_dict = load_data(args.data_folder)

    augmenter = ComposeAugmenters([
        Rotate(p=0.5, angle_range=180),
        TimeWarp(p=0.5, sigma=0.2, knot_range=4)
    ])

    train_sets = [ResampleArrayDataset(train_dict, augmenter=augmenter) for train_dict in train_dicts]
    valid_set = BasicArrayDataset(valid_dict)
    train_loaders = [DataLoader(train_set, batch_size=8, shuffle=True) for train_set in train_sets]
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    # create model
    backbone = TCN(
        input_shape=(200, 3),
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.8,
        use_spatial_dropout=True,
        conv_norm='batch',
        attention_conv_norm='batch'
    )
    classifier = MultiFCClassifiers(
        n_features=128,
        n_classes=[train_set.num_classes for train_set in train_sets]
    )
    model = CompleteModel(backbone=backbone, classifier=classifier, dropout=0.5)

    # create folder to save result
    save_folder = f'{args.output_folder}/{args.name}'
    last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
    last_run = max(last_run) + 1 if len(last_run) > 0 else 0
    save_folder = f'{save_folder}/run_{last_run}'

    # create training config
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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
        valid_loader=[valid_loader, None],  # only validate the first task
        num_epochs=num_epochs
    )

    train_log.to_csv(f'{save_folder}/train.csv', index=False)
    valid_log.to_csv(f'{save_folder}/valid.csv', index=False)

    print("Done!")
