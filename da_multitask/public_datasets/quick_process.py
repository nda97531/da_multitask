import os
import re
import numpy as np
from glob import glob
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from da_multitask.utils.dataframe import interpolate_numeric_df
from da_multitask.utils.sliding_window import sliding_window


class QuickProcess:
    def __init__(self, name: str, destination_folder: str, signal_freq: float = 50., window_size_sec: float = 4):
        """

        Args:
            name:
            destination_folder:
            signal_freq:
            window_size_sec: window size in second
        """
        self.name = name
        self.destination_folder = destination_folder
        # convert sec to num rows
        self.window_size_row = int(window_size_sec * signal_freq)
        # convert Hz to sample/msec
        self.signal_freq = signal_freq / 1000

    def _resample(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """

        Args:
            df:

        Returns:

        """
        start_ts = df.at[0, timestamp_col]
        end_ts = df.at[len(df) - 1, timestamp_col]

        # get new timestamp array (unit: msec)
        new_ts = np.arange(np.floor((end_ts - start_ts) * self.signal_freq + 1)) / self.signal_freq + start_ts
        new_ts = new_ts.astype(int)

        # interpolate
        df = interpolate_numeric_df(df, timestamp_col=timestamp_col, new_timestamp=new_ts)

        return df

    def run(self):
        raise NotImplementedError()


class URFall(QuickProcess):
    def __init__(self, raw_csv_folder: str,
                 name: str, destination_folder: str, signal_freq: float = 50., window_size_sec: float = 4):
        super().__init__(name, destination_folder, signal_freq, window_size_sec)
        self.raw_csv_folder = raw_csv_folder

    def _read_sequence(self, file) -> np.ndarray:
        """

        Args:
            file:

        Returns:

        """
        df = pd.read_csv(file, header=None, usecols=[0, 2, 3, 4])
        df.columns = ['msec', 'acc_x', 'acc_y', 'acc_z']
        # df = df.loc[df['msec'] >= 0].reset_index(drop=True)
        df = self._resample(df, timestamp_col='msec').to_numpy()
        if len(df) < self.window_size_row:
            pad_len = self.window_size_row - len(df)
            pad_ts = np.arange(pad_len)[::-1] / self.signal_freq
            df = np.pad(df, [[pad_len, 0], [0, 0]])
            df[:pad_len, 0] = df[pad_len, 0] - self.signal_freq ** -1 - pad_ts
        return df

    def _create_adl_windows(self, adl_files: list) -> np.ndarray:
        """

        Args:
            adl_files:

        Returns:

        """
        all_adl_windows = []
        for adl_file in adl_files:
            arr = self._read_sequence(adl_file)
            windows = sliding_window(arr, window_size=self.window_size_row,
                                     step_size=self.window_size_row // 2)
            all_adl_windows.append(windows)

        all_adl_windows = np.concatenate(all_adl_windows)
        return all_adl_windows

    def _create_fall_windows(self, fall_files: list) -> np.ndarray:
        """

        Args:
            fall_files:

        Returns:

        """
        all_fall_windows = []
        for fall_file in fall_files:
            arr = self._read_sequence(fall_file)
            window = arr[-self.window_size_row:]
            all_fall_windows.append(window)

        all_fall_windows = np.array(all_fall_windows)
        return all_fall_windows

    def run(self):
        # create Fall np array
        fall_files = glob(f'{self.raw_csv_folder}/fall*.csv')
        all_fall_windows = self._create_fall_windows(fall_files)

        # create ADL np array
        adl_files = glob(f'{self.raw_csv_folder}/adl*.csv')
        all_adl_windows = self._create_adl_windows(adl_files)

        print(all_adl_windows.shape, all_fall_windows.shape)

        os.makedirs(self.destination_folder, exist_ok=True)
        np.save(f'{self.destination_folder}/{self.name}_fall.npy', all_fall_windows)
        np.save(f'{self.destination_folder}/{self.name}_adl.npy', all_adl_windows)


class KFall(QuickProcess):
    FALL_TASK_ID = set('%02d' % n for n in range(20, 35))

    def __init__(self, raw_folder: str,
                 name: str, destination_folder: str, signal_freq: float = 50., window_size_sec: float = 4):
        super().__init__(name, destination_folder, signal_freq, window_size_sec)
        self.raw_data_folder = f'{raw_folder}/sensor_data'
        self.raw_label_folder = f'{raw_folder}/label_data'

    def _read_label(self, path: str) -> pd.DataFrame:
        """
        Read and modify label file

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

        Args:
            data_df:
            label_row:

        Returns:

        """
        # get label in msec
        fall_onset_frame = label_row.at['Fall_onset_frame']
        fall_impact_frame = label_row.at['Fall_impact_frame']
        frame_counter = data_df['FrameCounter'].to_numpy()
        fall_onset_msec = data_df.loc[(frame_counter == fall_onset_frame), 'msec'].iat[0]
        fall_impact_msec = data_df.loc[(frame_counter == fall_impact_frame), 'msec'].iat[0]
        assert fall_impact_msec > fall_onset_msec, 'fall_impact_msec must be > fall_onset_msec'

        # resample (indices change after this)
        data_df = self._resample(data_df[['msec', 'AccX', 'AccY', 'AccZ']], timestamp_col='msec')
        data_arr = data_df.to_numpy()

        # padding if not enough rows
        if len(data_arr) <= self.window_size_row:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            return window

        # find start & end indices by msec
        fall_indices = np.nonzero((data_arr[:, 0] <= fall_impact_msec) & (data_arr[:, 0] >= fall_onset_msec))[0]
        fall_onset_idx = fall_indices[0]
        fall_impact_idx = fall_indices[-1]
        num_needed_rows = self.window_size_row - (fall_impact_idx - fall_onset_idx + 1)
        expected_start_idx = fall_onset_idx - np.ceil(num_needed_rows / 2).astype(int)
        expected_end_idx = fall_impact_idx + np.floor(num_needed_rows / 2).astype(int)

        # if there's enough rows on both sides
        if expected_start_idx >= 0 and expected_end_idx < len(data_arr):
            window = data_arr[expected_start_idx:expected_end_idx + 1]
        # if there's not enough rows after fall
        elif expected_end_idx >= len(data_arr):
            window = data_arr[-self.window_size_row:]
        # if there's not enough rows before fall
        else:
            window = data_arr[:self.window_size_row]
        return window

    def _get_adl_windows(self, data_df: pd.DataFrame) -> np.ndarray:
        """

        Args:
            data_df:

        Returns:

        """
        data_df = self._resample(data_df[['msec', 'AccX', 'AccY', 'AccZ']], timestamp_col='msec')
        data_arr = data_df.to_numpy()
        if len(data_arr) >= self.window_size_row:
            windows = sliding_window(data_arr, window_size=self.window_size_row, step_size=self.window_size_row // 2)
        else:
            window = np.pad(data_arr, [[self.window_size_row - len(data_arr), 0], [0, 0]])
            windows = np.array([window])
        return windows

    def run(self):
        all_fall_windows = []
        all_adl_windows = defaultdict(list)

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
                    all_fall_windows.append(fall_window)
                # if this is an ADL session
                else:
                    adl_windows = self._get_adl_windows(session_data_df)
                    all_adl_windows[f'task{task_id}'].append(adl_windows)

        all_fall_windows = np.array(all_fall_windows)
        print(all_fall_windows.shape)

        os.makedirs(self.destination_folder, exist_ok=True)
        np.save(f'{self.destination_folder}/{self.name}_fall.npy', all_fall_windows)

        for task_id, task_data in all_adl_windows.items():
            task_data = np.concatenate(task_data)
            print(task_data.shape)
            np.save(f'{self.destination_folder}/{self.name}_{task_id}.npy', task_data)


class Museir(QuickProcess):
    """
    This dataset doesn't have fall data, so all are considered ADL
    """

    def __init__(self, processed_folder: str,
                 name: str, destination_folder: str, signal_freq: float = 50., window_size_sec: float = 4):
        super().__init__(name, destination_folder, signal_freq, window_size_sec)
        self.processed_folder = processed_folder

    def _read_sequence(self, path: str) -> pd.DataFrame:
        """

        Args:
            path:

        Returns:

        """
        df = pd.read_parquet(path, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z'])

        # convert m/s^2 -> g
        df[['acc_x', 'acc_y', 'acc_z']] /= 9.80665

        # resample
        df = self._resample(df, timestamp_col='timestamp')

        df['timestamp'] -= df.at[0, 'timestamp']
        return df

    def _get_windows_from_session(self, df: pd.DataFrame) -> np.ndarray:
        """

        Args:
            df:

        Returns:

        """

        windows = sliding_window(df.to_numpy(), window_size=self.window_size_row, step_size=self.window_size_row // 2)
        return windows

    def run(self):
        files = sorted(glob(f'{self.processed_folder}/setup_*/Phone_inertia/*.parquet'))
        all_adl_windows = []
        for df_file in files:
            df = self._read_sequence(df_file)
            windows = self._get_windows_from_session(df)
            all_adl_windows.append(windows)

        all_adl_windows = np.concatenate(all_adl_windows)

        print(all_adl_windows.shape)

        os.makedirs(self.destination_folder, exist_ok=True)
        np.save(f'{self.destination_folder}/Museir_adl.npy', all_adl_windows)


if __name__ == '__main__':
    # URFallProcess(
    #     raw_csv_folder='/mnt/data_drive/projects/datasets/UR Fall/raw/accelerometer/',
    #     name='URFall',
    #     destination_folder='./draft',
    #     signal_freq=50, window_size_sec=4
    # ).run()
    #
    KFall(
        raw_folder='/mnt/data_drive/projects/datasets/KFall/',
        name='KFall',
        destination_folder='./draft',
        signal_freq=50, window_size_sec=4
    ).run()

    # Museir(
    #     processed_folder='/mnt/data_drive/projects/UCD01 - Privacy preserving data collection/data/batch3/processed',
    #     name='Museir',
    #     destination_folder='./draft',
    #     signal_freq=50, window_size_sec=4
    # ).run()
