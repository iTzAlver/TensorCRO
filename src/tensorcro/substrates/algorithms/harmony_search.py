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