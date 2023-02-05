from typing import Union, List
import numpy as np


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

    def apply(self, org_data: np.ndarray) -> np.ndarray:
        """

        Args:
            org_data:

        Returns:

        """
        if (self.p >= 1) or (np.random.rand() < self.p):
            data = self._apply_logic(org_data)
            return data

        return org_data


class ComposeAugmenters(Augmenter):
    def __init__(self, augmenters: List[Augmenter], p: float = 1):
        super().__init__(p)
        self.augmenters = augmenters

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
        transpose_order = tuple(range(len(org_data.shape) - 2)) + (-1, -2)
        data = org_data.copy().transpose(transpose_order)

        # for every 3 channels
        for i in range(0, data.shape[-2], 3):
            data[..., i:i + 3, :] = self.rotate(data[..., i:i + 3, :], *rotate_angles)
        # transpose back to [time step, channel]
        data = data.transpose(transpose_order)
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

        # create rotation filters
        rotate_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        rotate_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        rotate_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])

        # rotate original data by multiply it with rotation filters
        rotate_filters = np.matmul(np.matmul(rotate_x, rotate_y), rotate_z)
        data = np.matmul(rotate_filters, data)
        return data


if __name__ == '__main__':
    aug = Rotate(p=1,
                 angle_x_range=[180, 180],
                 angle_y_range=[90, 90],
                 angle_z_range=0)

    data = aug.apply(np.ones([2, 10, 6]))
    print(data[0])
    print()
    print(data[1])
    print(data.shape)
