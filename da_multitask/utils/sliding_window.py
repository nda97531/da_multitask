import numpy as np


def sliding_window(data: np.ndarray, window_size: int, step_size: int, get_last: bool = True) -> np.ndarray:
    """

    Args:
        data:
        window_size:
        step_size:
        get_last:

    Returns:

    """
    num_windows = (len(data) - window_size) / step_size + 1
    if num_windows < 1:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)

    # if possible, run fast sliding window
    if window_size % step_size == 0:
        result = np.empty([int(num_windows), window_size, *data.shape[1:]], dtype=data.dtype)
        div = int(window_size / step_size)
        for window_idx, data_idx in enumerate(range(0, window_size, step_size)):
            new_window_data = data[data_idx:data_idx + (len(data) - data_idx) // window_size * window_size].reshape(
                [-1, window_size, *data.shape[1:]])

            new_window_idx = list(range(window_idx, int(num_windows), div))
            result[new_window_idx] = new_window_data
    # otherwise, run a regular loop
    else:
        result = np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)])

    if get_last and (num_windows % 1 != 0):
        result = np.concatenate([result, [data[-window_size:]]])

    return result
