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
class DifferentialSearch(CROSubstrate):
    def __init__(self,  directives: tf.Tensor, crossover_probability: float = 0.9, band_width: float = 0.8):
        """
        Differential Search Crossover Operator implements the Differential Evolution algorithm. It is a crossover
        operator that uses the following formula:

        .. math::
            x_i^{(k+1)} = x_i^{(k)} + F \cdot (x_{r_1}^{(k)} - x_{r_2}^{(k)}) \cdot \mathbb{1}_{\{r < CR\}}
        ..
        Where :math:`F` is the band-width, :math:`CR` is the crossover probability and :math:`r` is a random
        number between 0 and 1. The :math:`\mathbb{1}_{\{r < CR\}}` is a binary mask that is 1 if :math:`r < CR`
        and 0 otherwise.

        :param directives: Parameter specifications.
        :param crossover_probability: Crossover probability of the algorithm (CR).
        :param band_width: Band-width of the algorithm (F).
        """
        if not isinstance(directives, tf.Tensor):
            directives = tf.convert_to_tensor(directives)
        self.cr = tf.constant(crossover_probability)
        self.bw = tf.constant(band_width) * (directives[1] - directives[0])

    def _call(self, individuals: tf.Tensor):
        noi = tf.shape(individuals)[0]
        mask = tf.random.uniform([noi, tf.shape(individuals)[-1]], minval=0., maxval=1.)
        binary_mask = tf.less(mask, self.cr)
        random_individuals_diff = tf.random.shuffle(individuals) - tf.random.shuffle(individuals)
        random_individuals_2 = tf.random.shuffle(individuals)
        return tf.where(binary_mask, random_individuals_2 + self.bw * random_individuals_diff, individuals)

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
