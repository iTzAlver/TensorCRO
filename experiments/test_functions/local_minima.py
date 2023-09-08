# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from .optf import OptimizationFunction
import numpy as np


class AckleyFunction(OptimizationFunction):
    def __init__(self, a: float = 20., b: float = 0.2, c: float = 2. * np.pi):
        super().__init__()
        self.bounds = (-32.768, 32.768)
        self.a = a
        self.b = b
        self.c = c

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Ackley function.
        Formula:
            f(x) = -20 * exp(-0.2 * sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
            exp(0.5 * (cos(2 * pi * x[0]) + cos(2 * pi * x[1]))) + e + 20
        :param x: Input individual.
        :return:
        """
        a = self.a
        b = self.b
        c = self.c
        exp_1 = np.exp(-b * np.sqrt(np.mean(x ** 2, axis=-1)))
        exp_2 = np.exp(np.mean(np.cos(c * x), axis=-1))
        return -a * exp_1 - exp_2 + np.e + a


class BukinFunction6(OptimizationFunction):
    def __init__(self, scale: float = 1.):
        super().__init__()
        self.bounds = (-15. * scale, 3. * scale)
        self.scale = scale

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Bukin function.
        Formula:
            f(x) = 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)
        :param x: Input individual.
        :return:
        """
        a = 100 * self.scale
        b = a / 10000
        c = a / 10
        sqr = np.sqrt(np.mean(np.abs(x - b * np.expand_dims(x[:, 0], -1) ** 2), axis=-1))
        return a * sqr + b * np.abs(x[:, 0] + c)


class CrossInTrayFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-10., 10.)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Cross in Tray function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _k0 = -0.0001
        _sines = np.prod(np.sin(x), axis=-1)
        _exponent = np.exp(np.abs(100 - np.sqrt(np.sum(x ** 2)) / np.pi))
        return _k0 * ((np.abs(_sines * _exponent) + 1) ** 0.1)


class DropWaveFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Drop Wave function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _quads = np.mean(x ** 2, axis=-1)
        _num = 1 + np.cos(12 * np.sqrt(_quads) * np.sqrt(2))
        _den = _quads + 2
        return - _num / _den


class EggHolderFunction(OptimizationFunction):
    def __init__(self, k: int = 47):
        super().__init__()
        self.bounds = (-512, 512)
        self.k = k

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Egg Holder function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        k = self.k
        _slide = np.concatenate([np.zeros((np.shape(x)[0], 1)), x], axis=-1)[:, :-1] + k
        _first_member = x * np.sin(np.sqrt(np.abs(x - _slide)))
        _second_member = _slide * np.sin(np.sqrt(np.abs(x / 2 + _slide)))
        _combined_members = - (_first_member + _second_member)
        return np.sum(_combined_members, axis=-1)


class GramacyLeeFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (0.5, 2.5)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Gramacy - Lee's function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _each_x_sq = (np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4) ** 2
        return np.sum(_each_x_sq, axis=-1)


class GrieWankFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-600, 600)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Grie - Wank's function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _sum_sq = np.sum(x ** 2, axis=-1) / 4000
        _arrange_sqrt = np.sqrt(np.arange(1, np.shape(x)[1] + 1))
        _prod_cos = np.prod(np.cos(x / _arrange_sqrt), axis=-1)
        return _sum_sq - _prod_cos + 1


class HolderTableFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-10, 10)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Gramacy - Lee's function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _odds = np.sin(x[:, ::2])
        _even = np.cos(x[:, 1::2])
        _prod = np.prod(np.concatenate([_even, _odds], axis=-1), axis=-1)
        _exp = np.exp(np.abs(1 - np.sqrt(np.sum(x ** 2, axis=-1)) / np.pi))
        return - np.abs(_prod * _exp)


class RastriginFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Rastrigin function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _sum = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)
        return 10 * np.shape(x)[1] + _sum


class LevyFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Levy function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _w = 1 + (x - 1) / 4
        _first_member = np.sin(np.pi * _w[:, 0]) ** 2
        _second_member = np.sum((_w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * _w[:, :-1] + 1) ** 2), axis=-1)
        _third_member = (_w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * _w[:, -1]) ** 2)
        return _first_member + _second_member + _third_member


class ShubertFunction(OptimizationFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Shubert function.
        :param x: Input individual.
        :return: The value of each individual.
        """
        _i = np.arange(1, 6)
        _value = np.sum(_i * np.cos((_i + 1) * x[:, :, None] + _i), axis=-1)
        _return_value = np.prod(_value, axis=-1)
        return _return_value
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
