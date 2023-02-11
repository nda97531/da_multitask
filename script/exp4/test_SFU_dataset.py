import os
from glob import glob
import torch as tr
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from utils.sliding_window import sliding_window
from da_multitask.networks.backbone_tcn import TCN
from da_multitask.networks.classifier import MultiFCClassifiers, FCClassifier
from da_multitask.networks.complete_model import CompleteModel


def load_single_task_model(weight_path: str, device: str = 'cpu',
                           conv_norm='batch', attention_conv_norm='batch',
                           spatial_dopout=False) -> nn.Module:
    """

    Args:
        weight_path:

    Returns:

    """
    # create model
    backbone = TCN(
        input_shape=(200, 3),
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=spatial_dopout,
        conv_norm=conv_norm,
        attention_conv_norm=attention_conv_norm
    )
    classifier = FCClassifier(
        n_features=128,
        n_classes=2
    )
    model = CompleteModel(backbone=backbone, classifier=classifier).to(device)

    # load weight
    state_dict = tr.load(weight_path)
    model.load_state_dict(state_dict)
    return model


def load_multitask_model(weight_path: str, n_classes: list, device: str = 'cpu',
                         conv_norm='batch', attention_conv_norm='batch',
                         spatial_dopout=False) -> nn.Module:
    """

    Args:
        n_classes:
        weight_path:
        device:

    Returns:

    """
    # create model
    backbone = TCN(
        input_shape=(200, 3),
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 5 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=spatial_dopout,
        conv_norm=conv_norm,
        attention_conv_norm=attention_conv_norm
    )
    classifier = MultiFCClassifiers(
        n_features=128,
        n_classes=n_classes
    )
    model = CompleteModel(backbone=backbone, classifier=classifier).to(device)

    # load weight
    state_dict = tr.load(weight_path)
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
    label = int(label == 'Falls')
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

    print(classification_report(y_true, y_pred, digits=4))


def test_multitask(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files):
            data = get_windows_from_df(file).to(device)
            label = get_label_from_file_path(file)

            # only use the first task (index 0)
            pred = model(
                data,
                classifier_kwargs={'mask': tr.zeros(len(data), dtype=tr.int)}
            )[0]
            # predict fall (positive-1) if there's any positive window
            pred = pred.argmax(1).any().item()

            y_true.append(label)
            y_pred.append(pred)

    print(classification_report(y_true, y_pred, digits=4))


if __name__ == '__main__':
    print('compare norm layers, dropout layers')
    device = 'cuda:1'

    list_data_files = glob('/home/ducanh/projects/datasets/SFU/parquet/sub*/*/*.parquet')
    # weight_path_pattern = 'draft/exp4/run_{}/{}_task_last_epoch.pth'
    weight_path_pattern = 'draft/exp4/run_{}/{}_task.pth'

    # print('test run 3: D1 (2 classes of D1) and D1fall+D2 (22 classes of D1+D2), SGD momentum, spatial dropout')
    # model_3 = load_multitask_model(weight_path=weight_path_pattern.format(5, 'multi'),
    #                                n_classes=[2, 22], device=device,
    #                                spatial_dopout=True)
    # test_multitask(model_3, list_data_files)
    # del model_3

    print('test run 2: D1 (2 classes of D1) and D1fall+D2 (22 classes of D1+D2), SGD momentum, dropout')
    model_2 = load_multitask_model(weight_path=weight_path_pattern.format(4, 'multi'),
                                   n_classes=[2, 22], device=device,
                                   spatial_dopout=False)
    test_multitask(model_2, list_data_files)
    del model_2

    print('test run 1: D1 (2 classes of D1) and D1fall+D2 (22 classes of D1+D2), SGD, spatial dropout')
    model_1 = load_multitask_model(weight_path=weight_path_pattern.format(1, 'multi'),
                                   n_classes=[2, 22], device=device,
                                   spatial_dopout=True)
    test_multitask(model_1, list_data_files)
    del model_1

    print('test run 0: D1 (2 classes of D1) and D1fall+D2 (22 classes of D1+D2), SGD, dropout')
    model_0 = load_multitask_model(weight_path=weight_path_pattern.format(0, 'multi'),
                                   n_classes=[2, 22], device=device,
                                   spatial_dopout=False)
    test_multitask(model_0, list_data_files)
    del model_0
