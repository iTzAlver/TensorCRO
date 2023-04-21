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
class BLXAlphaCrossover(CROSubstrate):
    def __init__(self, directives: tf.Tensor, scale: float = 0.1):
        """
        The BLXAlphaCrossover method implements the BLX-Alpha crossover method. This method is a
        generalization of the BLX crossover method, which is a generalization of the SBX crossover
        method. The BLX-Alpha crossover method is a linear combination of the parents, with a
        random alpha value between 0 and 1.
        :param directives: Specifications of the parameters.
        :param scale: Scale of the alpha value.
        """
        self.mins = directives[0] * scale
        self.maxs = directives[1] * scale

    def _call(self, individuals: tf.Tensor):
        # Split the individuals in two groups
        noi = tf.shape(individuals)[0] // 2
        fathers = individuals[0:noi]
        mothers = individuals[noi:noi * 2]
        # Create the mask
        alpha = tf.random.uniform(shape=tf.shape(fathers), minval=self.mins, maxval=self.maxs, dtype=tf.float32)
        offspring1 = fathers + alpha * (mothers - fathers)
        offspring2 = mothers + alpha * (fathers - mothers)

        return tf.concat([offspring1, offspring2], axis=0)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
