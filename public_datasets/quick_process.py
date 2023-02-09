import os
import re

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import polars as pl
from collections import defaultdict
from tqdm import tqdm

from public_datasets.constant import G_TO_MS2
from utils.pd_dataframe import interpolate_numeric_df, write_df_file
from utils.sliding_window import sliding_window, shifting_window
from utils.time import TimeThis


class QuickProcess:
    def __init__(self, name: str, raw_folder: str, destination_folder: str,
                 signal_freq: float = 50., window_size_sec: float = 4):
        """
        This class transforms public datasets into the same format for ease of use.

        Args:
            name: name of the dataset
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            signal_freq: (Hz) resample signal to this frequency by linear interpolation
            window_size_sec: window size in second
        """
        self.name = name
        self.raw_folder = raw_folder
        self.destination_folder = destination_folder
        # pattern for output npy file name
        self.output_name_pattern = f'{destination_folder}/{name}_{{label}}/{{index}}.npy'

        # convert sec to num rows
        self.window_size_row = int(window_size_sec * signal_freq)
        # convert Hz to sample/msec
        self.signal_freq = signal_freq / 1000

    def resample(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Resample a dataframe

        Args:
            df: dataframe
            timestamp_col: name of the timestamp column

        Returns:
            resampled dataframe
        """
        start_ts = df.at[0, timestamp_col]
        end_ts = df.at[len(df) - 1, timestamp_col]

        # get new timestamp array (unit: msec)
        new_ts = np.arange(np.floor((end_ts - start_ts) * self.signal_freq + 1)) / self.signal_freq + start_ts
        new_ts = new_ts.astype(int)

        # interpolate
        df = interpolate_numeric_df(df, timestamp_col=timestamp_col, new_timestamp=new_ts)

        return df

    def write_npy_sequences(self, data: list, label: str):
        """
        Write all sequences into npy files

        Args:
            data: list of np arrays, each array is a sequence of shape [num windows, window length, 4(msec,x,y,z)]
            label: label of this data
        """
        num_digits = len(str(len(data)))
        index_pattern = f'%0{num_digits}d'

        for i, seq_windows in enumerate(data):
            output_path = self.output_name_pattern.format(label=label, index=index_pattern % i)
            os.makedirs(os.path.split(output_path)[0], exist_ok=True)
            print(f'writing {seq_windows.shape} to {output_path}')
            np.save(output_path, seq_windows)

    def run(self):
        """
        Main processing method
        """
        raise NotImplementedError()


class URFall(QuickProcess):
    def _read_sequence(self, file) -> np.ndarray:
        """

        Args:
            file:

        Returns:

        """
        df = pd.read_csv(file, header=None, usecols=[0, 2, 3, 4])
        df.columns = ['msec', 'acc_x', 'acc_y', 'acc_z']
        # df = df.loc[df['msec'] >= 0].reset_index(drop=True)
        df = self.resample(df, timestamp_col='msec').to_numpy()
        if len(df) < self.window_size_row:
            pad_len = self.window_size_row - len(df)
            pad_ts = np.arange(pad_len)[::-1] / self.signal_freq
            df = np.pad(df, [[pad_len, 0], [0, 0]])
            df[:pad_len, 0] = df[pad_len, 0] - self.signal_freq ** -1 - pad_ts
        return df

    def _create_adl_windows(self, adl_file: str) -> np.ndarray:
        """
        Turn an ADL DF file into a numpy array of data

        Args:
            adl_file: path to file

        Returns:
            numpy array shape [num windows, window length, channel]
        """
        arr = self._read_sequence(adl_file)
        windows = sliding_window(arr, window_size=self.window_size_row, step_size=self.window_size_row // 2)
        return windows

    def _create_fall_windows(self, fall_file: str) -> np.ndarray:
        """
        Turn a fall DF file into a numpy array of 1 fall window

        Args:
            fall_file: path to file

        Returns:
            numpy array shape [num windows, window length, 4(msec,x,y,z)]
        """
        arr = self._read_sequence(fall_file)
        window = arr[-self.window_size_row:]
        window = np.expand_dims(window, axis=0)
        return window

    def get_list_sequences(self, in_paths: list, label: str) -> list:
        """
        Read a list of dataframe files into a list of sequences

        Args:
            in_paths: list of input paths
            label: label of this data

        Returns:
            a list of sequences, each one is a numpy array of shape [num windows, window length, 4(msec,x,y,z)]
        """
        if label == 'fall':
            create_window_function = self._create_fall_windows
        elif label == 'adl':
            create_window_function = self._create_adl_windows
        else:
            raise ValueError('`label` is either fall or adl')

        sequences = []
        for i, file in enumerate(in_paths):
            seq_windows = create_window_function(file)
            sequences.append(seq_windows)
        return sequences

    def run(self):
        # create Fall np array
        fall_files = sorted(glob(f'{self.raw_folder}/fall*.csv'))
        print(f'Found {len(fall_files)} fall files of {self.name}')
        fall_sequences = self.get_list_sequences(fall_files, label='fall')
        self.write_npy_sequences(fall_sequences, label='fall')

        # create ADL np array
        adl_files = sorted(glob(f'{self.raw_folder}/adl*.csv'))
        print(f'Found {len(adl_files)} adl files of {self.name}')
        adl_sequences = self.get_list_sequences(adl_files, label='adl')
        self.write_npy_sequences(adl_sequences, label='adl')


class KFall(QuickProcess):
    FALL_TASK_ID = set('%02d' % n for n in range(20, 35))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_data_folder = f'{self.raw_folder}/sensor_data'
        self.raw_label_folder = f'{self.raw_folder}/label_data'
        # minimum step size between fall windows
        self.min_fall_window_step_size = self.window_size_row // 2
        # make sure at least 1s of post-fall impact event is included in the window
        self.expand_after_impact = int(self.signal_freq * 1000)

    def _read_label(self, path: str) -> pd.DataFrame:
        """
        Read label file

        Args:
            path: path to file

        Returns:
            dataframe of labels
        """
        df = pd.read_excel(path)
        df['Task Code (Task ID)'] = df['Task Code (Task ID)'].map(
            lambda s: int(re.match(r'(?:\s+)?F(?:[0-9]+) \(([0-9]+)\)(?:\s+)?', s).group(1))
            if pd.notna(s) else None
        )
        df = df.fillna(method='ffill')
        df = df.rename({'Task Code (Task ID)': 'Task ID'}, axis=1)
        return df

    def _read_data(self, path: str) -> pd.DataFrame:
        """
        Read data file without interpolation because it must be unchanged for label matching

        Args:
            path: path to data csv file

        Returns:
            dataframe of sensor data
        """
        df = pd.read_csv(path, usecols=['TimeStamp(s)', 'FrameCounter', 'AccX', 'AccY', 'AccZ'])
        df['TimeStamp(s)'] *= 1000
        df = df.rename({'TimeStamp(s)': 'msec'}, axis=1)
        return df

    def _get_session_info(self, session_id: str) -> tuple:
        """
        Get subject ID, task ID, trial ID from session ID

        Args:
            session_id: session ID (name of data file)

        Returns:
            (subject ID, task ID, trial ID), all are strings
        """
        res = re.match(r'S([0-9]+)T([0-9]+)R([0-9]+)', session_id)
        subject_id, task_id, trial_id = [res.group(i) for i in range(1, 4)]
        return subject_id, task_id, trial_id

    def _get_fall_window(self, data_df: pd.DataFrame, label_row: pd.Series) -> np.ndarray:
        """
        Turn a fall session DF into a numpy array of 1 fall window

        Args:
            data_df: data df
            label_row: a row in the label df, label of this data df

        Returns:
            numpy array shape [num windows, window length, 4(msec,x,y,z)]
        """
        # get label in msec
        fall_onset_frame = label_row.at['Fall_onset_frame']
        fall_impact_frame = label_row.at['Fall_impact_frame']
        frame_counter = data_df['FrameCounter'].to_numpy()
        fall_onset_msec = data_df.loc[(frame_counter == fall_onset_frame), 'msec'].iat[0]
        fall_impact_msec = data_df.loc[(frame_counter == fall_impact_frame), 'msec'].iat[0]
        assert fall_impact_msec > fall_onset_msec, 'fall_impact_msec must be > fall_onset_msec'

        # resample (indices change after this)
        data_df = self.resample(data_df[['msec', 'AccX', 'AccY', 'AccZ']], timestamp_col='msec')
        data_arr = data_df.to_numpy()

        # padding if not enough rows
        if len(data_arr) <= self.window_size_row:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            return np.expand_dims(window, axis=0)

        # find start & end indices by msec
        fall_indices = np.nonzero((data_arr[:, 0] <= fall_impact_msec) & (data_arr[:, 0] >= fall_onset_msec))[0]
        fall_onset_idx = fall_indices[0]
        fall_impact_idx = fall_indices[-1]

        windows = shifting_window(
            data_arr,
            window_size=self.window_size_row, max_num_windows=3, min_step_size=self.min_fall_window_step_size,
            start_idx=fall_onset_idx, end_idx=fall_impact_idx + self.expand_after_impact
        )
        return windows

    def _get_adl_windows(self, data_df: pd.DataFrame) -> np.ndarray:
        """

        Args:
            data_df:

        Returns:

        """
        data_df = self.resample(data_df[['msec', 'AccX', 'AccY', 'AccZ']], timestamp_col='msec')
        data_arr = data_df.to_numpy()
        if len(data_arr) >= self.window_size_row:
            windows = sliding_window(data_arr, window_size=self.window_size_row, step_size=self.window_size_row // 2)
        else:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            windows = np.array([window])
        return windows

    def run(self):
        all_fall_sequences = []
        all_adl_sequences = defaultdict(list)

        # go through all sequence files
        for subject_id in tqdm(sorted(os.listdir(self.raw_data_folder))):
            label_df = self._read_label(f'{self.raw_label_folder}/{subject_id}_label.xlsx')

            for session_file in sorted(glob(f'{self.raw_data_folder}/{subject_id}/*.csv')):
                session_data_df = self._read_data(session_file)
                subject_id, task_id, trial_id = self._get_session_info(session_file.split('/')[-1][:-4])

                # if this is a fall session
                if task_id in self.FALL_TASK_ID:
                    label_row = label_df.loc[(label_df['Task ID'] == int(task_id)) &
                                             (label_df['Trial ID'] == int(trial_id))]
                    assert label_row.shape[0] == 1, f'Something is wrong with session {session_file}'
                    fall_window = self._get_fall_window(session_data_df, label_row.iloc[0])
                    all_fall_sequences.append(fall_window)
                # if this is an ADL session
                else:
                    adl_windows = self._get_adl_windows(session_data_df)
                    all_adl_sequences[f'task{task_id}'].append(adl_windows)

        # write fall data
        self.write_npy_sequences(all_fall_sequences, label='fall')
        # write adl data
        for task_id, task_data in all_adl_sequences.items():
            self.write_npy_sequences(task_data, label=task_id)


class UCISmartphone(QuickProcess):

    def _read_label_file(self, path: str):
        """

        Args:
            path:

        Returns:

        """
        df = pd.read_csv(path, header=None, sep=' ')
        df.columns = ['exp', 'user', 'activity_id', 'start_idx', 'end_idx']

        # change start and end idx if signal freq is not 50Hz (0.05 sample/msec)
        if self.signal_freq != 0.05:
            df[['start_idx', 'end_idx']] = (df[['start_idx', 'end_idx']] / 0.05 * self.signal_freq).round().astype(int)

        return df

    def _read_data_file(self, path: str):
        """

        Args:
            path:

        Returns:

        """
        df = pd.read_csv(path, header=None, sep=' ')
        df.columns = ['acc_x', 'acc_y', 'acc_z']
        # this dataset is 50Hz (interval 20 msec)
        df['msec'] = np.arange(len(df)) * 20
        # resample if the requirement is not 50Hz (0.05 sample/msec)
        if self.signal_freq != 0.05:
            df = self.resample(df, timestamp_col='msec')

        df = df[['msec', 'acc_x', 'acc_y', 'acc_z']]
        return df

    def _get_session_info(self, data_file_name: str) -> tuple:
        """

        Args:
            data_file_name:

        Returns:

        """
        match = re.match(f'acc_exp([0-9][0-9])_user([0-9][0-9]).txt', data_file_name)
        exp_id = match.group(1)
        user_id = match.group(2)
        return exp_id, user_id

    def run(self):
        accel_files = sorted(glob(f'{self.raw_folder}/acc_*.txt'))
        all_label_df = self._read_label_file(f'{self.raw_folder}/labels.txt')

        all_sequences = []
        for file in accel_files:
            data_df = self._read_data_file(file)

            # remove rows that are not in labelled range
            exp_id, user_id = self._get_session_info(file.split('/')[-1])
            label_df = all_label_df.loc[
                (all_label_df['exp'] == int(exp_id)) & (all_label_df['user'] == int(user_id))]
            first_idx = label_df['start_idx'].iat[0]
            last_idx = label_df['end_idx'].iat[-1]
            data_arr = data_df.to_numpy()[first_idx:last_idx + 1]

            # sliding window
            data_arr = sliding_window(data_arr, window_size=self.window_size_row, step_size=self.window_size_row // 2)
            all_sequences.append(data_arr)

        self.write_npy_sequences(all_sequences, label='adl')


class MobiActV2(QuickProcess):
    FALL_LABELS = np.array(['FOL', 'FKL', 'BSC', 'SDL'])
    STATIC_LABELS = np.array(['STD', 'SIT', 'LYI'])
    RAW_FREQUENCY = 200  # Hz

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.RAW_FREQUENCY % int(self.signal_freq * 1000) == 0, \
            'This dataset can only be resampled by arranging, not interpolation, ' \
            'so raw frequency must be divisible by target frequency'
        self.downsample_by = int(self.RAW_FREQUENCY / int(self.signal_freq * 1000))
        # minimum step size between fall windows
        self.min_fall_window_step_size = self.window_size_row // 4

    def read_sequence_file(self, path: str) -> np.ndarray:
        """
        Read a csv data file

        Args:
            path: path to file

        Returns:
            dataframe with 4 columns [msec, acc_x, acc_y, acc_z, label]
        """
        df = pl.read_csv(path)
        df = df.select([
            pl.col('timestamp').alias('msec'),
            pl.col('acc_x'), pl.col('acc_y'), pl.col('acc_z'),
            pl.col('label')
        ]).with_columns([
            (pl.col('msec') / 1e6).cast(int),  # nano-sec -> msec
            pl.col('acc_x') / G_TO_MS2,
            pl.col('acc_y') / G_TO_MS2,
            pl.col('acc_z') / G_TO_MS2
        ]).to_numpy()
        # resample
        df = df[np.arange(0, len(df), self.downsample_by)]
        return df

    def get_label_data_dict(self, arr: np.ndarray) -> dict:
        """
        Turn a data array into a dict

        Args:
            arr: array shape [n, 5(msec, acc x, acc y, acc z, label]

        Returns:
            a dict:
                - key: label name
                - value: array shape [num windows, window size, 4(msec,x,y,z)]
        """
        # if this is a fall sequence
        is_fall_label = np.isin(arr[:, -1], self.FALL_LABELS)
        if is_fall_label.any():
            # padding if not enough rows
            if len(arr) <= self.window_size_row:
                window = np.pad(arr, [[self.window_size_row - len(arr), 0], [0, 0]])
                return {'fall': np.expand_dims(window, axis=0)}

            # find fall start/end indices
            is_fall_label = np.concatenate([[False], is_fall_label, [False]])
            fall_start_end_idxs = np.diff(is_fall_label).nonzero()[0]
            fall_start_end_idxs[1::2] -= 1
            fall_start_end_idxs = fall_start_end_idxs.reshape([-1, 2])

            # shifting fall windows
            sequence_fall_windows = []
            for fall_start_idx, fall_end_idx in fall_start_end_idxs:
                fall_windows = shifting_window(
                    arr,
                    window_size=self.window_size_row, max_num_windows=3, min_step_size=self.min_fall_window_step_size,
                    start_idx=fall_start_idx, end_idx=fall_end_idx
                )
                sequence_fall_windows.append(fall_windows)
            sequence_fall_windows = np.concatenate(sequence_fall_windows)
            return {'fall': sequence_fall_windows}

        # if this is a non-fall sequence
        windows = sliding_window(arr, self.window_size_row, self.window_size_row // 2)

        # find label for each window
        labels = []
        for label in windows[:, :, -1].astype('U'):
            value, count = np.unique(label, return_counts=True)
            argmax = np.argmax(count)
            value = value[argmax]
            count = count[argmax]

            # only count static activities if they occupy the whole window
            if value in self.STATIC_LABELS and count < len(label):
                labels.append('')
            else:
                labels.append(value)

        labels = np.array(labels)

        # create dict
        windows = windows[:, :, :-1].astype(float)
        label_data_dict = {label: windows[labels == label] for label in np.unique(labels) if label != ''}

        return label_data_dict

    def run(self):
        files = sorted(glob(f'{self.raw_folder}/*/*.csv'))
        label_data_dict = defaultdict(list)

        for file in tqdm(files):
            arr = self.read_sequence_file(file)
            label_data_dict_of_file = self.get_label_data_dict(arr)
            for key, value in label_data_dict_of_file.items():
                label_data_dict[key].append(value)

        for label, list_sequences in label_data_dict.items():
            self.write_npy_sequences(list_sequences, label)


class SFI(QuickProcess):
    """
    This dataset is processed differently from the others because it is used as test set.
    Its label doesn't suit to be a training set.
    (A long sequence has only 1 label, and fall can be anywhere within. Though peak can be used if needed)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = 'sub{subject_id}/{label}/{filename}'

    def _read_sequence(self, excel_file: str) -> pd.DataFrame:
        """

        Args:
            excel_file:

        Returns:

        """
        df = pl.read_excel(excel_file)
        df = df.select([
            pl.col('Time').alias('msec'),
            pl.col('waist Acceleration X (m/s^2)').alias('acc_x'),
            pl.col('waist Acceleration Y (m/s^2)').alias('acc_y'),
            pl.col('waist Acceleration Z (m/s^2)').alias('acc_z')
        ]).with_columns([
            (pl.col('msec') / 1000).cast(int),
            pl.col('acc_x') / G_TO_MS2,
            pl.col('acc_y') / G_TO_MS2,
            pl.col('acc_z') / G_TO_MS2
        ]).to_pandas()
        df = self.resample(df, timestamp_col='msec')
        return df

    def _write_parquet(self, df: pd.DataFrame, excel_path: str):
        """

        Args:
            df:
            excel_path:

        Returns:

        """
        child_path = excel_path.removeprefix(self.raw_folder).removesuffix('.xlsx')
        save_path = f'{self.destination_folder}/{child_path}.parquet'
        write_df_file(df, save_path)

    def run(self):
        files = glob(f'{self.raw_folder}/{self.pattern.format(subject_id="*", label="*", filename="*")}')
        for file in tqdm(files):
            df = self._read_sequence(file)
            self._write_parquet(df, file)


if __name__ == '__main__':
    pass
    # URFall(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/UR Fall/raw/accelerometer/',
    #     name='URFall',
    #     destination_folder='/mnt/data_drive/projects/npy_data_seq',
    #     signal_freq=50, window_size_sec=4
    # ).run()

    # KFall(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/KFall/',
    #     name='KFall',
    #     destination_folder='/mnt/data_drive/projects/npy_data_seq',
    #     signal_freq=50, window_size_sec=4
    # ).run()

    # UCISmartphone(
    #     raw_folder='/mnt/data_drive/projects/raw datasets/uci-smartphone-based-recognition-of-human-activities/RawData/',
    #     name='UCISmartphone',
    #     destination_folder='/mnt/data_drive/projects/npy_data_seq',
    #     signal_freq=50, window_size_sec=4
    # ).run()

    MobiActV2(
        raw_folder='/mnt/data_drive/projects/raw datasets/MobiAct2/MobiAct_Dataset_v2.0/Annotated Data',
        name='MobiActV2',
        destination_folder='/mnt/data_drive/projects/npy_data_seq',
        signal_freq=50, window_size_sec=4
    ).run()

    # unlike the others, SFI is written as parquet and no sliding window because it is the test set
    # SFI(
    #     raw_folder='/mnt/data_drive/projects/datasets/SFU-IMU Dataset/raw',
    #     name='SFI',
    #     destination_folder='/mnt/data_drive/projects/datasets/SFU-IMU Dataset/parquet',
    #     signal_freq=50  # , window_size_sec=4
    # ).run()
