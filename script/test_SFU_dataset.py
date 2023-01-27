import os
from glob import glob
import torch as tr
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils.sliding_window import sliding_window
from da_multitask.networks.backbone import TCN
from da_multitask.networks.classifier import MultiFCClassifiers, FCClassifier
from da_multitask.networks.complete_model import CompleteModel


def load_single_task_model(weight_path: str, device: str = 'cpu') -> nn.Module:
    """

    Args:
        weight_path:

    Returns:

    """
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
    model = CompleteModel(backbone=backbone, classifier=classifier).to(device)

    # load weight
    state_dict = tr.load(weight_path).to(device)
    model.load_state_dict(state_dict)


def load_multitask_model(weight_path: str, device: str = 'cpu') -> nn.Module:
    """

    Args:
        weight_path:
        device:

    Returns:

    """
    # create model
    backbone = TCN(
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=False
    )
    classifier = MultiFCClassifiers(
        n_features=128,
        n_classes=[2, 21]
    )
    model = CompleteModel(backbone=backbone, classifier=classifier).to(device)

    # load weight
    state_dict = tr.load(weight_path).to(device)
    model.load_state_dict(state_dict)

    return model


def get_windows_from_df(file: str) -> tr.Tensor:
    df = pd.read_parquet(file)
    arr = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    windows = sliding_window(arr, window_size=200, step_size=100)
    windows = tr.from_numpy(windows).float()
    return windows


def get_label_from_file_path(path: str) -> int:
    """

    Args:
        path:

    Returns:

    """
    label = path.split(os.sep)[-2]
    label = (label == 'Falls')
    return label


def test_single_task(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files):
            data = get_windows_from_df(file).to(device)
            label = get_label_from_file_path(file)

            # predict fall (positive-1) if there's any positive window
            pred = model(data).argmax(1).any().item()

            y_true.append(label)
            y_pred.append(pred)

    print(classification_report(y_true, y_pred))


def test_multitask(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files):
            data = get_windows_from_df(file).to(device)
            label = get_label_from_file_path(file)

            # only use the first task (index 0)
            pred = model(data, mask=tr.zeros(len(data), dtype=int))[0]
            # predict fall (positive-1) if there's any positive window
            pred = pred.argmax(1).any().item()

            y_true.append(label)
            y_pred.append(pred)

    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    list_data_files = glob('/mnt/data_drive/projects/datasets/SFU-IMU Dataset/parquet/sub*/*/*.parquet')
    weight_path_pattern = '/mnt/data_drive/projects/other code/da_multitask/draft/exp_{}/single_task_last_epoch.pth'

    # test model 0: single task, train on D1
    model_0 = load_single_task_model(weight_path=weight_path_pattern.format(0), device='cpu')
    test_single_task(model_0, list_data_files)
    del model_0

    # test model 1: single task, train on D1+D2
    model_1 = load_single_task_model(weight_path=weight_path_pattern.format(1), device='cpu')
    test_single_task(model_1, list_data_files)
    del model_1

    # test model 2: multitask, train on D1 and D2
    model_2 = load_multitask_model(weight_path=weight_path_pattern.format(2), device='cpu')
    test_multitask(model_2, list_data_files)
