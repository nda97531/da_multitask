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


def load_single_task_model(weight_path: str, device: str = 'cpu') -> nn.Module:
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
        use_spatial_dropout=False,
        conv_norm='batch',
        attention_conv_norm='batch'
    )
    classifier = FCClassifier(
        n_features=128,
        n_classes=1
    )
    model = CompleteModel(backbone=backbone, classifier=classifier).to(device)

    # load weight
    state_dict = tr.load(weight_path)
    model.load_state_dict(state_dict)
    return model


def load_multitask_model(weight_path: str, n_classes: list, device: str = 'cpu') -> nn.Module:
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
        use_spatial_dropout=False,
        conv_norm='batch',
        attention_conv_norm='batch'
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
        for file in tqdm(list_data_files, ncols=0):
            data = get_windows_from_df(file).to(device)
            label = get_label_from_file_path(file)

            # predict fall (positive-1) if there's any positive window
            pred = model(data).squeeze(1)
            assert len(pred.shape) == 1
            pred = (tr.sigmoid(pred) > 0.5).any().item()

            y_true.append(label)
            y_pred.append(pred)

    print(classification_report(y_true, y_pred, digits=4))
    return classification_report(y_true, y_pred, output_dict=True)


def test_multitask(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files, ncols=0):
            data = get_windows_from_df(file).to(device)
            label = get_label_from_file_path(file)

            # only use the first task (index 0)
            pred = model(
                data,
                classifier_kwargs={'mask': tr.zeros(len(data), dtype=tr.int)}
            )[0].squeeze(1)
            # predict fall (positive-1) if there's any positive window
            assert len(pred.shape) == 1
            pred = (tr.sigmoid(pred) > 0.5).any().item()

            y_true.append(label)
            y_pred.append(pred)

    print(classification_report(y_true, y_pred, digits=4))
    return classification_report(y_true, y_pred, output_dict=True)


if __name__ == '__main__':
    device = 'cuda:1'

    list_data_files = glob('/home/ducanh/projects/datasets/SFU/parquet/sub*/*/*.parquet')
    # weight_path_pattern = 'draft/{exp_id}/run_{run_id}/{task}_task_last_epoch.pth'
    weight_path_pattern = 'draft/result_exp7_bceloss/{exp_id}/run_{run_id}/{task}_task.pth'

    # key: exp id; value: a dict of {precision, recall, f1score}
    all_results = {}

    # region: test single task
    exps = [
        'g1.1',
        'g1.2'
    ]
    for exp_id in exps:
        print(f'------------------------------\ntesting exp {exp_id}')

        exp_results = []
        for run_id in range(3):
            model = load_single_task_model(
                weight_path=weight_path_pattern.format(exp_id=exp_id, run_id=run_id, task='single'),
                device=device
            )

            result = test_single_task(
                model=model,
                list_data_files=list_data_files,
                device=device
            )
            exp_results.append(result['1'])
        
        assert exp_id not in all_results
        all_results[exp_id] = pd.DataFrame(exp_results).mean(axis=0)
    # endregion
    
    # region: test multi task
    num_classes = {
        'g2.1': [1, 21],
        # 'g2.2': [2, 11],
        # 'g2.3': [2, 21, 11],
        # 'g3.1': [2, 21],
        # 'g3.2': [2, 23],
        'g3.3': [1, 22],
        'g3.4': [1, 1]
    }
    for exp_id, num_class in num_classes.items():
        print(f'------------------------------\ntesting exp {exp_id}')

        exp_results = []
        for run_id in range(3):
            model = load_multitask_model(
                weight_path=weight_path_pattern.format(exp_id=exp_id, run_id=run_id, task='multi'),
                n_classes=num_class,
                device=device
            )
            result = test_multitask(
                model=model,
                list_data_files=list_data_files,
                device=device
            )
            exp_results.append(result['1'])

        assert exp_id not in all_results
        all_results[exp_id] = pd.DataFrame(exp_results).mean(axis=0)
    # endregion

    all_results = pd.DataFrame(all_results).transpose().round(decimals=4)
    print(all_results)
