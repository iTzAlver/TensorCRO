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
class MaskedCrossover(CROSubstrate):
    def __init__(self, mask: list):
        """
        The MaskedCrossover method implements a mask for the crossover method. The mask is a list of 0s and 1s, where
        0s indicate that the corresponding gene will be taken from the father, and 1s indicate that the corresponding
        gene will be taken from the mother.
        :param mask: Mask for the crossover method.
        """
        self.mask = tf.convert_to_tensor(mask)

    def _call(self, individuals: tf.Tensor):
        # Split the individuals in two groups
        noi = tf.shape(individuals)[0] // 2
        fathers = individuals[0:noi]
        mothers = individuals[noi:noi * 2]
        # Create the mask
        extended_mask = tf.tile([tf.equal(self.mask, 0)], [noi, 1])
        offspring1 = tf.where(extended_mask, fathers, mothers)
        offspring2 = tf.where(extended_mask, mothers, fathers)

        return tf.concat([offspring1, offspring2], axis=0)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
