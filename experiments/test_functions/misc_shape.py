# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from .optf import OptimizationFunction


class StyblinskiTang(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5., 5.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Styblinski-Tang function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        x_4 = np.power(x, 4)
        x_2 = np.power(x, 2)
        return np.sum(x_4 - 16 * x_2 + 5 * x, axis=1) / 2


class Powell(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-4., 5.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Powell function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _how_many_zeros_are_left = (4 - x.shape[1] % 4) % 4
        _zeros = np.zeros((x.shape[0], _how_many_zeros_are_left))
        _x = np.concatenate((x, _zeros), axis=1)
        set_0 = _x[:, 0::4]
        set_1 = _x[:, 1::4]
        set_2 = _x[:, 2::4]
        set_3 = _x[:, 3::4]
        _1 = np.power(set_0 + 10 * set_1, 2)
        _2 = 5 * np.power(set_2 - set_3, 2)
        _4 = 10 * np.power(set_0 - set_3, 4)
        _3 = np.power(set_1 - 2 * set_2, 4)

        return np.sum(_1 + _2 + _3 + _4, axis=1) / 2

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
