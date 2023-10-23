# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import tensorflow as tf
from ..substrate import CROSubstrate


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class GaussianCrossover(CROSubstrate):
    def __init__(self, mean: float = 0.75, stddev: float = 0.25):
        """
        The GaussianCrossover method implements a gaussian crossover. It is a selector between two parents,
        where the probability of selecting a parent is given by a gaussian distribution. If the mean is 0.5 it is
        equivalent to a uniform crossover.
        :param mean: Mean of the gaussian.
        :param stddev: Standard deviation of the gaussian.
        """
        self.mean = tf.constant(mean)
        self.stddev = tf.constant(stddev)

    def _call(self, individuals: tf.Tensor, **kwargs):
        noi = tf.shape(individuals)[0] // 2
        fathers = individuals[0:noi]
        mothers = individuals[noi:noi * 2]
        selector = tf.cast(tf.round(tf.random.normal(shape=tf.shape(fathers), mean=self.mean,
                                                     stddev=self.stddev, dtype=tf.float32)),
                           dtype=tf.bool)
        return tf.where(selector, fathers, mothers)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
