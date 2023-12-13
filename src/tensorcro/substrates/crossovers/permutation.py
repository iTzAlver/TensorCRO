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
class RandomPermutation(CROSubstrate):
    def __init__(self, section_points: int):
        """
        This substrate performs a random permutation of the parameters. It swaps the parameters in the
        same section.
        :param section_points: Section points of the parameters.
        """
        self.section_points = tf.constant(section_points, dtype=tf.int32)

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        # Empty vector of shape [0, 2 * nturbines]:
        permuted_shaped = None
        # Permute individuals, so we have a 2D tensor with each individual and its permuted parameters:
        multip = tf.shape(individuals)[-1] // self.section_points
        for individual in individuals:
            # We create a permutation tensor:
            permutation_tensor = tf.random.shuffle(tf.range(multip))
            permutation_tensor = tf.cast(permutation_tensor, tf.float32)
            permutation_tensor = tf.expand_dims(permutation_tensor, axis=0)
            permutation_tensor = tf.tile(permutation_tensor, [self.section_points, 1])
            # Reshape to [1, -1]:
            permutation_tensor = tf.cast(tf.reshape(permutation_tensor, [1, -1]), tf.int32)
            # Add [0, 0, 0, ...] until tf.shape[-1] // 2 and [1, 1, 1, 1, ...] until from tf.shape[-1] //
            # 2 to tf.shape[-1]:
            permutations_add = tf.concat([tf.zeros(shape=(1, multip), dtype=tf.int32), multip * tf.ones(
                shape=(1, multip), dtype=tf.int32)], axis=-1)
            # We add the values:
            permutations = permutation_tensor + permutations_add
            # We permute the individual:
            individual_out_unshaped = tf.gather(individual, permutations, axis=-1)
            # We append the individual to the list:
            if permuted_shaped is not None:
                permuted_shaped = tf.concat([permuted_shaped, individual_out_unshaped], axis=0)
            else:
                permuted_shaped = individual_out_unshaped
        # Remove all the permutated elements that are the same that in the original individual:
        repeated_index = tf.reduce_all(tf.equal(permuted_shaped, individuals), axis=-1)
        # Choose only false values:
        repeated_index = tf.logical_not(repeated_index)
        # Get indices where true:
        repeated_index = tf.where(repeated_index)
        # We filter the individuals:
        retval = tf.gather(individuals, repeated_index, axis=0)
        # We return the individuals:
        return tf.squeeze(retval)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
