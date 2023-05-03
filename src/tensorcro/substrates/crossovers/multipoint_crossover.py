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
class MultipointCrossover(CROSubstrate):
    def __init__(self, points: list):
        """
        The MultipointCrossover class implements a multipoint crossover. The points is a list of indices where
        the points between the parents are swapped. The mask is applied to each parent, so the offspring will be
        composed of the points of the first parent in the positions indicated by the mask, and the points of the
        second parent in the positions not indicated by the mask.
        :param points: List of indices where the points between the parents are swapped.
        """
        self.points = tf.convert_to_tensor(points)

    def _call(self, individuals: tf.Tensor):
        # Split the individuals in two groups
        noi = tf.shape(individuals)[0] // 2
        fathers = individuals[0:noi]
        mothers = individuals[noi:noi * 2]

        # Generate the mask
        oned = tf.scatter_nd(tf.expand_dims(self.points, 1), tf.ones_like(self.points), shape=[tf.shape(fathers)[1]])
        tiled_oned = tf.tile([oned], multiples=[noi, 1])
        mask = tf.cast(tf.cumsum(tiled_oned, axis=1) % 2, dtype=tf.bool)

        offspring1 = tf.where(mask, fathers, mothers)
        offspring2 = tf.where(mask, mothers, fathers)

        return tf.concat([offspring1, offspring2], axis=0)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
