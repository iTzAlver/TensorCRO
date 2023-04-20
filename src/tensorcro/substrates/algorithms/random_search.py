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
        self.size = size
        self.params = directives.shape[-1]
        self.min = directives[0]
        self.max = directives[1]

    def _call(self, individuals: tf.Tensor):
        if self.size <= 1:
            noi = tf.round(tf.cast(tf.shape(individuals)[0], dtype=tf.float32) * self.size)
        else:
            noi = tf.round(self.size)
        shape = tf.concat([[noi], tf.shape(individuals)[1:]], axis=0)
        return tf.random.uniform(shape=shape, minval=self.min, maxval=self.max, dtype=tf.float32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
