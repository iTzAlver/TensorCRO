# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from .optf import OptimizationFunction


class Easom(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-100., 100.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Easom function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        cosined_member = np.cos(x)
        exp_member = -((x - np.pi) ** 2)
        cosined = np.prod(cosined_member, -1)
        exp = np.exp(np.sum(exp_member, -1))
        return -cosined * exp


class Michalewicz(OptimizationFunction):
    def __init__(self, m: float = 10):
        super().__init__()
        self.m = m
        self.bounds = (0., np.pi)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Michalewicz function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        sined = np.sin(x)
        insine = np.sin((np.arange(1, x.shape[-1] + 1) * x ** 2) / np.pi)
        expsine = insine ** (2 * self.m)
        return -np.sum(sined * expsine, -1)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
