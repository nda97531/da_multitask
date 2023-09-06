from glob import glob
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
    state_dict = tr.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_windows_from_df(file: str) -> tuple:
    # this dataset doesn't have fall activity
    df = pl.read_parquet(file)
    arr = df.select(['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)']).to_numpy()
    arr /= G_TO_MS2

    data_windows = sliding_window(arr, window_size=200, step_size=200)
    data_windows = data_windows.astype(np.float32) 
    return tr.from_numpy(data_windows)


def cal_score(y_pred_prob: np.ndarray):
    """
    Because the RealWorld dataset doesn't have any fall activity, all y_trues are 0.
    Calculate metrics: specificity

    Args:
        y_pred_prob: 1d array of prediction probability

    Returns:
        a dict with keys are metric names
    """
    # fall: True; non-fall: False
    y_pred = y_pred_prob > 0.5
    
    # specificity
    true_neg = (~y_pred).sum()
    specificity = true_neg / len(y_pred)

    result = {'specificity': specificity, 'support': len(y_pred)}
    print(result)
    return result


def test_single_task(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)
    y_pred_prob = []
    with tr.no_grad():
        for file in tqdm(list_data_files, ncols=0):
            data = get_windows_from_df(file).to(device)

            pred = model(data).squeeze(1)
            pred = tr.sigmoid(pred).cpu()
            y_pred_prob.append(pred)

    y_pred_prob = np.concatenate(y_pred_prob)
    result = cal_score(y_pred_prob)
    return result


def test_multitask(model: nn.Module, list_data_files: list, device: str = 'cpu'):
    model = model.eval().to(device)
    y_pred_prob = []
    with tr.no_grad():
        for file in tqdm(list_data_files, ncols=0):
            data = get_windows_from_df(file).to(device)

            # only use the first task (index 0)
            pred = model(
                data,
                classifier_kwargs={'mask': tr.zeros(len(data), dtype=tr.int)}
            )[0].squeeze(1)
            pred = tr.sigmoid(pred).cpu()
            y_pred_prob.append(pred)

    y_pred_prob = np.concatenate(y_pred_prob)
    result = cal_score(y_pred_prob)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device

    list_data_files = glob('/home/ducanh/parquet_datasets/RealWorld/inertia/subject_*/*.parquet')
    weight_path_pattern = ('/home/ducanh/projects/UCD02 - Multitask Fall det/da_multitask/log'
                           '/{exp_id}/run_{run_id}/{task}_task.pth')

    # key: exp id; value: a dict of {precision, recall, f1score}
    all_results = {}

    # region: test single task
    exps = [
        'g1_1',
        'g1_2'
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
        # result = pd.DataFrame(exp_results).mean(axis=0)
        result = pd.DataFrame(exp_results)
        result = result.iloc[np.argmax(result['specificity'])]
        all_results[exp_id] = result
    # endregion
    
    # region: test multi task
    num_classes = {
        'g2_1': [1, 12],
        'g3_1': [1, 14],
        'g3_2': [1, 13],
        'g3_3': [1, 12],
        'g3_4': [1, 13],
        'g3_5': [1, 1],
        'g3_6': [1, 1],
        'g3_7': [1, 14],
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
        # result = pd.DataFrame(exp_results).mean(axis=0)
        result = pd.DataFrame(exp_results)
        result = result.iloc[np.argmax(result['specificity'])]
        all_results[exp_id] = result
    # endregion

    all_results = pd.DataFrame(all_results).transpose().round(decimals=4)
    print(all_results)
