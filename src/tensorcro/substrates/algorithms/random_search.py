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
class RandomSearch(CROSubstrate):
    def __init__(self, directives: tf.Tensor, size: float = .5):
        """
        This class implements the Random Search algorithm. It is a simple algorithm that generates random individuals
        within the search space.
        :param directives: Parameter specifications.
        :param size: Size of the new population (float or int). If float, it is the percentage of the original
        population and if int, it is the number of new individuals.
        """
        if not isinstance(directives, tf.Tensor):
            directives = tf.convert_to_tensor(directives)
        self.size = size
        self.params = directives.shape[-1]
        self.min = directives[0]
        self.max = directives[1]

    def _call(self, individuals: tf.Tensor, **kwargs):
        if self.size <= 1:
            noi = tf.round(tf.cast(tf.shape(individuals)[0], dtype=tf.float32) * self.size)
        else:
            noi = tf.round(self.size)
        shape = tf.concat([[noi], tf.shape(individuals)[1:]], axis=0)
        return tf.random.uniform(shape=shape, minval=self.min, maxval=self.max, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
