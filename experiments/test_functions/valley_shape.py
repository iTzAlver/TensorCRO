# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from .optf import OptimizationFunction


class Rosenbrock(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5., 10.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Rosenbrock function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        rolled = np.roll(x, -1, axis=1)
        rolled[:, -1] = 0
        results = 100 * (x - rolled ** 2) ** 2 + (rolled - 1) ** 2
        return np.sum(results, axis=-1)


class DixonPrice(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-10., 10.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Dixon Price function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        first_member = (x[:, 0] - 1) ** 2
        second_member = (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2
        i_vector = np.arange(2, x.shape[1] + 1)
        third_member = np.sum(i_vector * second_member)
        result = first_member + third_member
        return result
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
