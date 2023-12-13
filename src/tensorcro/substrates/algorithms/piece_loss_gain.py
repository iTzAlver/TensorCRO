# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from ..substrate import CROSubstrate


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class PieceSubstrate(CROSubstrate):
    def __init__(self, bounds: tf.Tensor, alpha: float = 1., nmute: float = 0.1, loss: bool = True):
        """
        This substrate is used to reduce or augment the energy of the individuals.
        -> Starts at alpha%.
        -> Increases to [-alpha to alpha] depending on the correlation of the individuals.
        :param alpha: Decrement value.
        :param nmute: Number of parameters to reduce in mean.
        :param bounds: The bounds of the individuals.
        :param loss: Whether to add or subtract the mutation.
        """
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.nmute = tf.constant(nmute / tf.shape(bounds[-1]), dtype=tf.float32)
        self.bounds = bounds
        if loss:
            self.operation = tf.subtract
        else:
            self.operation = tf.add

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        # Create a matrix with ones with mean nmean ones and variance nvar ones:
        mask = tf.random.uniform(shape=(tf.shape(individuals)[0], tf.shape(individuals)[1]),
                                 minval=0, maxval=1, dtype=tf.float32) < self.nmute
        ones = tf.ones(shape=(tf.shape(individuals)[0], tf.shape(individuals)[1]), dtype=tf.float32)
        zeros = tf.zeros(shape=(tf.shape(individuals)[0], tf.shape(individuals)[1]), dtype=tf.float32)
        subs = tf.where(mask, ones, zeros)

        # Get the negative mutation:
        new_individuals_unclipped = self.operation(individuals, subs)

        # Clip the individuals:
        new_individuals = tf.clip_by_value(new_individuals_unclipped, self.bounds[0], self.bounds[1])

        # Return the new individuals:
        return new_individuals
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
