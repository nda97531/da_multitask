from typing import Union, List
import numpy as np
from transforms3d.axangles import axangle2mat
from utils.number_array import gen_random_curves


def format_range(x: any, start_0: bool) -> np.ndarray:
    """
    Turn an arbitrary input into a range. For example:
        1 to [-1, 1] or [0, 1]
        None to [0, 0]
        [-1, 1] will be kept intact

    Args:
        x: any input
        start_0: if input x is a scalar and this is True, the starting point of the range will be 0,
            otherwise it will be -abs(x)

    Returns:
        a np array of 2 element [range start, range end]
    """
    if x is None:
        x = [0, 0]
    if isinstance(x, float) or isinstance(x, int):
        x = abs(x)
        x = [0 if start_0 else -x, x]
    assert len(x) == 2, 'x must be a scalar or a list/tuple of 2 numbers'
    return np.array(x)


class Augmenter:
    def __init__(self, p: float):
        """

        Args:
            p: range (0, 1], possibility to apply this augmenter each time `apply` is called
        """
        assert 0 < p <= 1
        self.p = p

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        """

        Args:
            org_data:

        Returns:

        """
        raise NotImplementedError()

    def _copy_data(self, org_data: np.ndarray) -> np.ndarray:
        """

        Args:
            org_data:

        Returns:

        """
        return org_data.copy()

    def apply(self, org_data: np.ndarray) -> np.ndarray:
        """

        Args:
            org_data: [time step, channel]

        Returns:

        """
        if (self.p >= 1) or (np.random.rand() < self.p):
            data = self._copy_data(org_data)
            data = self._apply_logic(data)
            return data

        return org_data


class ComposeAugmenters(Augmenter):
    def __init__(self, augmenters: List[Augmenter], p: float = 1):
        super().__init__(p)
        self.augmenters = augmenters

    def _copy_data(self, org_data: np.ndarray) -> np.ndarray:
        """
        No copy needed because all elements augmenters do this
        """
        return org_data

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        for i in np.random.permutation(range(len(self.augmenters))):
            org_data = self.augmenters[i].apply(org_data)
        return org_data


class Rotate(Augmenter):
    def __init__(self, p: float, angle_range: Union[list, tuple, float] = None) -> None:
        """
        Rotate tri-axial data in a random axis.

        Args:
            p: probability to apply this augmenter each time it is called
            angle_range: (degree) the angle is randomised within this range;
                if this is a list, randomly pick an angle in this range;
                if it's a float, the range is [-float, float]
        """
        super().__init__(p=p)

        self.angle_range = format_range(angle_range, start_0=False) / 180 * np.pi

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        """
        Apply augmentation methods in self.list_aug_func
        :param org_data:
            shape (time step, channel) channel must be divisible by 3,
            otherwise bugs may occur
        :return: array shape (time step, channel)
        """
        assert (len(org_data.shape) >= 2) and (org_data.shape[-1] % 3 == 0), \
            f"expected data shape: [*, any length, channel%3==0], got {org_data.shape}"

        angle = np.random.uniform(low=self.angle_range[0], high=self.angle_range[1])
        direction_vector = np.random.uniform(-1, 1, size=3)

        # transpose data to shape [channel, time step]
        data = org_data.T

        # for every 3 channels
        for i in range(0, data.shape[-2], 3):
            data[i:i + 3, :] = self.rotate(data[i:i + 3, :], angle, direction_vector)

        # transpose back to [time step, channel]
        data = data.T
        return data

    @staticmethod
    def rotate(data, angle: float, axis: np.ndarray):
        """
        Rotate data array

        Args:
            data: data array, shape [3, n]
            angle: a random angle in radian
            axis: a 3-d vector, the axis to rotate around

        Returns:
            rotated data of the same format as the input
        """
        rot_mat = axangle2mat(axis, angle)
        data = np.matmul(rot_mat, data)
        return data


class TimeWarp(Augmenter):
    def __init__(self, p: float, sigma: float = 0.2, knot_range: Union[int, list] = 4):
        """

        Args:
            p:
            sigma:
            knot_range:
        """
        super().__init__(p)
        self.sigma = sigma
        self.knot_range = format_range(knot_range, start_0=True)
        # add one here because upper bound is exclusive when randomising
        self.knot_range[1] += 1

    def distort_time_steps(self, length: int, num_curves: int):
        """

        Args:
            length:
            num_curves:
        Returns:

        """
        knot = np.random.randint(self.knot_range[0], self.knot_range[1])
        tt = gen_random_curves(length, num_curves, self.sigma, knot)
        tt_cum = np.cumsum(tt, axis=0)

        # Make the last value equal length
        t_scale = (length - 1) / tt_cum[-1]

        tt_cum *= t_scale
        return tt_cum

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        # create new timestamp for all channels
        tt_new = self.distort_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[:, i]) for i in range(org_data.shape[-1])
        ]).T
        return data
