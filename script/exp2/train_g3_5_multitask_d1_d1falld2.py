import os
import torch
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import defaultdict

from da_multitask.data_generator.augment import Augmenter, ComposeAugmenters, Rotate, TimeWarp
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
    train_dict_2 = {0: [], 1: []}  # D1fall+D2, 2 D1 classes
    valid_dict = {0: [], 1: []}

    # GET D2
    files = sorted(glob(f'{folder}/D2/*/*.npy'))
    print(f'{len(files)} files found for D2')
    for file in files:
        arr = np.load(file)[:, :, 1:]
        train_dict_2[0].append(arr)

    # GET D1, both train and valid
    files = sorted(glob(f'{folder}/D1/*/*.npy'))
    print(f'{len(files)} files found for D1')

    # read all files in D1
    d1_windows_train = []
    d1_labels_train = []
    d1_windows_valid = []
    d1_labels_valid = []
    for i, file in enumerate(files):
        arr = np.load(file)[:, :, 1:]

        file_label = int(file.split('/')[-2].endswith('_fall'))

        if i % 2 != 0:
            d1_windows_valid.append(arr)
            d1_labels_valid += [file_label] * len(arr)
        else:
            d1_windows_train.append(arr)
            d1_labels_train += [file_label] * len(arr)

    d1_windows_train = np.concatenate(d1_windows_train)
    d1_labels_train = np.array(d1_labels_train)

    d1_windows_valid = np.concatenate(d1_windows_valid)
    d1_labels_valid = np.array(d1_labels_valid)

    # append D1train into train_dict(s)
    train_dict_1[0].append(d1_windows_train[d1_labels_train == 0])
    train_dict_1[1].append(d1_windows_train[d1_labels_train == 1])
    train_dict_2[1].append(d1_windows_train[d1_labels_train == 1])

    # append D1valid into valid dict
    valid_dict[0].append(d1_windows_valid[d1_labels_valid == 0])
    valid_dict[1].append(d1_windows_valid[d1_labels_valid == 1])

    # return result
    train_dict_1 = {key: np.concatenate(value) for key, value in train_dict_1.items()}
    train_dict_2 = {key: np.concatenate(value) for key, value in train_dict_2.items()}
    valid_dict = {key: np.concatenate(value) for key, value in valid_dict.items()}
    return [train_dict_1, train_dict_2], valid_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='cuda:0')
    parser.add_argument('--name', '-n', required=True,
                        help='name of the experiment to create a folder to save weights')
    parser.add_argument('--data-folder', '-data',
                        default='/home/ducanh/projects/UCD02 - Multitask Fall det/npy_data',
                        help='path to data folder')
    parser.add_argument('--output-folder', '-o', default='./log',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    num_epochs = 300
    patience = 30

    train_dicts, valid_dict = load_data(args.data_folder)

    # train 3 times
    for _ in range(3):
        # create data loaders

        augmenter = Rotate(p=1, angle_range=180)

        train_sets = [ResampleArrayDataset(deepcopy(train_dict), augmenter=augmenter) for train_dict in train_dicts]
        valid_set = BasicArrayDataset(deepcopy(valid_dict))
        train_loaders = [DataLoader(train_set, batch_size=16, shuffle=True) for train_set in train_sets]
        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)

        # create model
        backbone = TCN(
            input_shape=[valid_dict[0].shape[1], 3],
            how_flatten='spatial attention gap',
            n_tcn_channels=(64,) * 5 + (128,) * 2,
            tcn_drop_rate=0.5,
            use_spatial_dropout=True,
            conv_norm='batch',
            attention_conv_norm='batch'
        )
        classifier = MultiFCClassifiers(
            n_features=128,
            n_classes=[train_set.num_classes if train_set.num_classes > 2 else 1 for train_set in train_sets]
        )
        model = CompleteModel(backbone=backbone, classifier=classifier, dropout=0.5)

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        loss_fn = 'classification_auto'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        flow = TrainFlow(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(num_epochs, f'{save_folder}/multi_task.pth'),
                EarlyStop(patience),
                # ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=lr_patience, verbose=True)
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
