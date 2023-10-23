# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import abc
import tensorflow as tf


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class CROSubstrate:
    @abc.abstractmethod
    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        return individuals

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return self._call(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__


class ComposedSubstrate(CROSubstrate):
    def __init__(self, *subs, name: str = 'ComposedSubstrate'):
        """
        The ComposedSubstrate method implements the CRO substrates as a composition of other substrates,
        in sequential order. The output of each substrate is the input of the next one.
        :param subs: Substrates to be composed.
        :param name: Name of the composed substrate.
        """
        self.subs = subs
        self.__name__ = name

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        _individuals = individuals
        for sub in self.subs:
            _individuals = sub(_individuals, **kwargs)
        return _individuals

    def __repr__(self):
        return self.__name__
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
