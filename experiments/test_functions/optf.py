# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import numpy as np
from abc import ABC, abstractmethod
MAXIMUM_OUTPUT = 1_000_000_000
MINIMUM_OUTPUT = -MAXIMUM_OUTPUT


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class OptimizationFunction(ABC):
    def __init__(self):
        self.number_of_evaluations = 0

    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        return x

    def __call__(self, *args, **kwargs):
        """
        Call the function.
        :param args: The parameters of the function.
        :param kwargs: Ignored.
        :return: The result of the function.
        """
        self.number_of_evaluations += len(args)
        retval = self.function(*args)
        clipped_retval = np.clip(retval, MINIMUM_OUTPUT, MAXIMUM_OUTPUT)
        return clipped_retval
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
