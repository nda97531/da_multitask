from typing import Union, List
import numpy as np

from utils.number_array import gen_random_curves


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
        for aug in self.augmenters:
            org_data = aug.apply(org_data)
        return org_data


class Rotate(Augmenter):
    def __init__(self, p: float,
                 angle_x_range: Union[list, tuple, float] = None,
                 angle_y_range: Union[list, tuple, float] = None,
                 angle_z_range: Union[list, tuple, float] = None) -> None:
        """

        Args:
            p:
            angle_x_range: if this is a list, randomly pick an angle in this range;
                if it's a float, the range is [-float, float]
            angle_y_range: same as `angle_x_range`
            angle_z_range: same as `angle_x_range`
        """
        super().__init__(p=p)

        def format_input(x):
            if x is None:
                x = [0, 0]
            if isinstance(x, float) or isinstance(x, int):
                x = abs(x)
                x = [-x, x]
            assert len(x) == 2, 'x must be a scalar or a list/tuple of 2 numbers'
            # convert angles from angle to radian
            x = np.array(x) / 180. * np.pi
            return x

        self.angle_range = list(zip(
            format_input(angle_x_range),
            format_input(angle_y_range),
            format_input(angle_z_range)
        ))

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        """
        Apply augmentation methods in self.list_aug_func
        :param org_data:
            shape (time step, channel) channel must be divisible by 3,
            otherwise bugs may occur
        :return: array shape (time step, channel)
        """
        assert (len(org_data.shape) >= 2) and (org_data.shape[-1] % 3 == 0), \
            "expected data shape: [*, any length, channel%3==0]"

        rotate_angles = np.random.uniform(low=self.angle_range[0], high=self.angle_range[1])

        # transpose data to shape [*, channel, time step]
        data = org_data.T

        # for every 3 channels
        for i in range(0, data.shape[-2], 3):
            data[..., i:i + 3, :] = self.rotate(data[..., i:i + 3, :], *rotate_angles)
        # transpose back to [time step, channel]
        data = data.T
        return data

    @staticmethod
    def rotate(data, rotate_x, rotate_y, rotate_z):
        """
        Rotate an array

        Args:
            data: shape (*, 3, time step)
            rotate_x: angle in RADIAN
            rotate_y: angle in RADIAN
            rotate_z: angle in RADIAN

        Returns:
            array shape (*, 3, time step)
        """
        cos_x = np.cos(rotate_x)
        sin_x = np.sin(rotate_x)
        cos_y = np.cos(rotate_y)
        sin_y = np.sin(rotate_y)
        cos_z = np.cos(rotate_z)
        sin_z = np.sin(rotate_z)

        rotate_filters = np.array([
            [cos_y * cos_z, sin_x * sin_y * cos_z - cos_x * sin_z, cos_x * sin_y * cos_z + sin_x * sin_z],
            [cos_y * sin_z, sin_x * sin_y * sin_z + cos_x * cos_z, cos_x * sin_y * sin_z - sin_x * cos_z],
            [-sin_y, sin_x * cos_y, cos_x * cos_y]
        ])

        data = np.matmul(rotate_filters, data)
        return data


class TimeWarp(Augmenter):
    def __init__(self, p: float, sigma: float = 0.2, knot: int = 4):
        """

        Args:
            p:
            sigma:
            knot:
        """
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def distort_time_steps(self, length: int, num_curves: int):
        """

        Args:
            length:
            num_curves:
        Returns:

        """
        tt = gen_random_curves(length, num_curves, self.sigma, self.knot)
        tt_cum = np.cumsum(tt, axis=0)

        # Make the last value equal length
        t_scale = (length - 1) / tt_cum[-1]

        tt_cum *= t_scale
        return tt_cum

    def _copy_data(self, org_data: np.ndarray) -> np.ndarray:
        # don't need to copy because `_apply_logic` doesn't modify org_data
        return org_data

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        # create new timestamp for all channels
        tt_new = self.distort_time_steps(org_data.shape[-2], 1).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[:, i]) for i in range(org_data.shape[-1])
        ]).T
        return data
