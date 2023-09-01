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

    # GET D1, both train and valid
    files = sorted(glob(f'{folder}/D1/*/*.npy'))
    print(f'{len(files)} files found for D1')

    # read all files in D1
    d1_windows = []
    d1_labels = []
    for file in files:
        arr = np.load(file)[:, :, 1:]
        file_label = int(os.path.split(file)[0].endswith('_fall'))
        d1_windows.append(arr)
        d1_labels += [file_label] * len(arr)
    d1_windows = np.concatenate(d1_windows)
    d1_labels = np.array(d1_labels)

    # split D1 into train and valid
    train_idx = np.arange(len(d1_windows)) % 2 != 0
    d1_windows_train = d1_windows[train_idx]
    d1_labels_train = d1_labels[train_idx]
    d1_windows_valid = d1_windows[~train_idx]
    d1_labels_valid = d1_labels[~train_idx]

    # append D1train into train_dict(s)
    train_dict[0].append(d1_windows_train[d1_labels_train == 0])
    train_dict[1].append(d1_windows_train[d1_labels_train == 1])

    # append D1valid into valid dict
    valid_dict[0].append(d1_windows_valid[d1_labels_valid == 0])
    valid_dict[1].append(d1_windows_valid[d1_labels_valid == 1])

    # return result
    train_dict = {key: np.concatenate(value) for key, value in train_dict.items()}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}
    
    return train_dict, valid_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, required=True)
    parser.add_argument('--name', '-n', default='g1.1',
                        help='name of the experiment to create a folder to save weights')
    parser.add_argument('--data-folder', '-data', default='/home/ducanh/projects/npy_data_seq/',
                        help='path to data folder')
    parser.add_argument('--output-folder', '-o', default='./draft',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    # train 3 times
    for _ in range(3):
        # create data loaders
        train_dict, valid_dict = load_data(args.data_folder)

        augmenter = ComposeAugmenters(
            p=1,
            augmenters=[
                Rotate(p=0.5, angle_range=180),
                # TimeWarp(p=0.5, sigma=0.2, knot_range=4)
            ]
        )

        train_set = ResampleArrayDataset(train_dict, augmenter=augmenter)
        valid_set = BasicArrayDataset(valid_dict)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)

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
            n_classes=1
        )
        model = CompleteModel(backbone=backbone, classifier=classifier, dropout=0.5)

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        loss_fn = 'classification_auto'
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
