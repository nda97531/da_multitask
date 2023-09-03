from glob import glob
from scipy.stats import mode
import numpy as np
import torch as tr
import torch.nn as nn
import pandas as pd
import polars as pl
from sklearn import metrics
from tqdm import tqdm
from da_multitask.utils.sliding_window import sliding_window, shifting_window
from da_multitask.networks.backbone_tcn import TCN
from da_multitask.networks.classifier import MultiFCClassifiers, FCClassifier
from da_multitask.networks.complete_model import CompleteModel
from da_multitask.public_datasets.constant import G_TO_MS2


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
    state_dict = tr.load(weight_path, map_location=device)
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


def get_windows_from_df(file: str) -> tuple:
    df = pl.read_parquet(file)
    arr = df.select(['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)', 'label']).to_numpy()
    arr[:, :3] /= G_TO_MS2

    # if this is a fall session:
    if arr[:, -1].any():
        fall_idx = np.nonzero(arr[:, -1])[0]
        windows = shifting_window(arr, window_size=200, max_num_windows=5,
                                  min_step_size=20, start_idx=fall_idx[0], end_idx=fall_idx[-1])
        data_windows = windows[:, :, :-1].astype(np.float32)
        label_windows = np.ones(len(data_windows), dtype=int)
    else:
        windows = sliding_window(arr, window_size=200, step_size=100)
        data_windows = windows[:, :, :-1].astype(np.float32)
        label_windows = np.zeros(len(data_windows), dtype=int)

    # windows = sliding_window(arr, window_size=200, step_size=100)
    # data_windows = windows[:, :, :-1].astype(np.float32)
    # label_windows = windows[:, :, -1].astype(int)
    # label_windows = mode(label_windows, axis=-1, nan_policy='raise', keepdims=False).mode

    return tr.from_numpy(data_windows), label_windows


def cal_score(y_true: np.ndarray, y_pred_prob: np.ndarray):
    """
    Calculate metrics: precision, recall, f1-score, auc-roc, auc-pr of the positive class

    Args:
        y_true: 1d array of binary label (0 or 1)
        y_pred_prob: 1d array of prediction probability

    Returns:
        a dict with keys are metric names
    """
    # precision, recall, f1
    y_pred = (y_pred_prob > 0.5).astype(int)
    result = metrics.classification_report(y_true, y_pred, output_dict=True)['1']

    # auc-roc
    result['auc-roc'] = metrics.roc_auc_score(y_true, y_pred_prob)

    # auc-pr
    precision_y, recall_x, thresholds = metrics.precision_recall_curve(y_true, y_pred_prob)
    result['auc-pr'] = metrics.auc(recall_x, precision_y)

    print(metrics.classification_report(y_true, y_pred, digits=4))
    print(result)
    return result


def test_single_task(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files, ncols=0):
            data, label = get_windows_from_df(file)

            # predict fall (positive-1) if there's any positive window
            pred = model(data.to(device)).squeeze(1)
            pred = tr.sigmoid(pred).cpu()

            y_true.append(label)
            y_pred.append(pred)
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    result = cal_score(y_true, y_pred)
    return result


def test_multitask(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    with tr.no_grad():
        for file in tqdm(list_data_files, ncols=0):
            data, label = get_windows_from_df(file)

            # only use the first task (index 0)
            pred = model(
                data.to(device),
                classifier_kwargs={'mask': tr.zeros(len(data), dtype=tr.int)}
            )[0].squeeze(1)
            pred = tr.sigmoid(pred).cpu()

            y_true.append(label)
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    result = cal_score(y_true, y_pred)
    return result


if __name__ == '__main__':
    device = 'cuda:0'

    list_data_files = glob('/home/ducanh/parquet_datasets/FallAllD/inertia/subject_*/*waist*.parquet')
    weight_path_pattern = ('/home/ducanh/projects/UCD02 - Multitask Fall det/da_multitask/log'
                           '/{exp_id}/run_{run_id}/{task}_task.pth')

    # key: exp id; value: a dict of {precision, recall, f1score}
    all_results = {}

    # region: test single task
    exps = [
        'g1_1_spatialdr',
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
            exp_results.append(result)

        assert exp_id not in all_results
        all_results[exp_id] = pd.DataFrame(exp_results).mean(axis=0)
    # endregion

    # region: test multi task
    num_classes = {
        # 'g2.1': [1, 21],
        # 'g3.1': [1, 23],
        # 'g3.2': [1, 22],
        # 'g3.3': [1, 21],
        # 'g3.4': [1, 22],
        # 'g3.5': [1, 1],
        # 'g3.6': [1, 1],
        # 'g3.7': [1, 23],
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
            exp_results.append(result)

        assert exp_id not in all_results
        all_results[exp_id] = pd.DataFrame(exp_results).mean(axis=0)
    # endregion

    all_results = pd.DataFrame(all_results).transpose().round(decimals=4)
    print(all_results)
