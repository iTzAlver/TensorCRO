# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import tensorflow as tf
from ..substrate import CROSubstrate
from .random_search import RandomSearch
from ..mutation import Mutation


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class HarmonySearch(CROSubstrate):
    def __init__(self, directives: tf.Tensor, hmc_r: float = 0.95, pa_r: float = 0.8, bandwidth: float = 0.2):
        """
        Harmony Search Crossover Operator implements the Harmony Search algorithm. It is a crossover operator that uses
        the following formula:

        .. math::
            x_i^{(k+1)} =

                x_i^{(k)} & if r_1 < HMC_{r}


                x_{r_2}^{(k)} & if r_1 >= HMC_{r} \text{ and } r_2 < PA_{r}


                x_i^{(k)} + BW \cdot (U_{min} - U_{max}) & if  r_1 >= HMC_{r}  and  r_2 \geq PA_{r}
        ..
        Where :math:`BW` is the band-width, :math:`HMC_{r}` is the Harmony Memory Consideration rate, :math:`PA_{r}` is
        the Pitch Adjustment rate and :math:`r_1` and :math:`r_2` are random numbers between 0 and 1.

        :param directives: Parameter specifications.
        :param hmc_r: The Harmony Memory Consideration rate (HMCR).
        :param pa_r: The Pitch Adjustment rate (PAR).
        :param bandwidth: The band-width (BW).
        """
        if not isinstance(directives, tf.Tensor):
            directives = tf.convert_to_tensor(directives)
        self.hmc_r = tf.constant(hmc_r)
        self.pa_r = tf.constant(pa_r)
        self.bw = tf.constant(bandwidth) * (directives[1] - directives[0])
        self.rs = RandomSearch(directives, 1)
        self.mut = Mutation('uniform', minval=directives[0], maxval=directives[1])

    def _call(self, individuals: tf.Tensor):
        noi = tf.shape(individuals)[0]
        mask = tf.random.uniform([2 * noi, tf.shape(individuals)[-1]], minval=0., maxval=1.)
        binary_mask_0 = tf.less(mask[:noi], self.hmc_r)
        binary_mask_1 = tf.less(mask[noi:], self.pa_r)
        random_individuals = self.rs(individuals)
        uniforms = tf.add(individuals, self.bw * self.mut(individuals))
        return tf.where(binary_mask_1, tf.where(binary_mask_0, individuals, random_individuals), uniforms)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
