# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from .optf import OptimizationFunction


class PermZDB(OptimizationFunction):
    def __init__(self, dims: int = 2, beta: float = 0.5):
        """
        Perm. Zero D beta function class.
        :param dims: Number of dimensions.
        :param beta: The beta parameter.
        """
        super().__init__()
        self.bounds = (-dims, dims)
        self.beta = beta

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Perm. Zero D beta function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        dims = np.shape(x)[-1]
        j = np.arange(dims)
        x_pow = np.expand_dims(x[:, j], -1) ** (j + 1)

        _j = np.arange(1, dims + 1).reshape(-1, 1)
        _i = np.arange(1, dims + 1)
        inverse = _j ** _i
        inner_0 = (j + 1 + self.beta) * x_pow / inverse
        sum_0 = np.pow(np.sum(inner_0, axis=-2), 2)
        result = np.sum(sum_0, axis=-1)
        return result


class HyperEpsiloid(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-65.536, 65.536)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        HyperEpsiloid function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        x_sq = x ** 2
        dim_vec = np.arange(np.shape(x)[-1], 0, -1)
        return np.sum(x_sq * dim_vec, axis=-1)


class HyperSphere(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        HyperSphere function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        return np.sum(x ** 2, axis=-1)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
