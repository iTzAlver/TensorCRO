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
class EnergyReductionSubstrate(CROSubstrate):
    def __init__(self, alpha: float = 0.1, likelihood: float = 1 - tf.math.sqrt(0.5)):
        """
        This substrate is used to reduce the energy of the individuals. It is a stateful algorithm that
        keeps track of the energy of the individuals and reduces it by 1% each time it is called depending on
        the correlation of all the individuals.
        -> Starts at alpha%.
        -> Increases to [-alpha to alpha] depending on the correlation of the individuals.
        :param alpha: The initial value of the decrement.
        :param likelihood: The likelihood of the individuals to be repeated.
        """
        self.base_alpha = tf.constant(alpha, dtype=tf.float32)
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32)
        self.likelihood = tf.Variable(initial_value=likelihood, dtype=tf.float32)

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        # Get the cross correlation:
        cross_correlation = tf.matmul(individuals, tf.transpose(individuals))
        # Maximum of the correlation:
        normalization_term = tf.math.reduce_max([tf.math.reduce_max(cross_correlation), 0.001])
        # Normalize the correlation:
        correlation = tf.math.divide(cross_correlation, normalization_term)
        # Get the new likelihood:
        new_likelihood = tf.abs(tf.math.reduce_mean(correlation))
        # Get the new alpha = alpha + alpha0 * ([0, 1] - likelihood):
        new_a = tf.clip_by_value(tf.math.add(self.alpha, tf.math.multiply(self.base_alpha,
                                                                          (new_likelihood -
                                                                           self.likelihood))),
                                 0, 1)
        self.alpha.assign(new_a)
        self.likelihood.assign(new_likelihood)

        # Get random mask to be added to the individuals with alpha probability:
        mask = tf.random.uniform(shape=tf.shape(individuals), minval=0, maxval=1) < self.alpha

        # Get the negative mutation:
        negative_mutation = tf.multiply(individuals,
                                        (1 - tf.abs(tf.random.normal(shape=tf.shape(individuals),
                                                                     mean=0, stddev=new_a))))
        # Get the new individuals:
        new_individuals = tf.where(mask, individuals, negative_mutation)
        return new_individuals


class EnergyAugmentationSubstrate(CROSubstrate):
    def __init__(self, alpha: float = 0.1, likelihood: float = 1 - tf.math.sqrt(0.5)):
        """
        This substrate is used to augment the energy of the individuals. It is a stateful algorithm that
        keeps track of the energy of the individuals and reduces it by 1% each time it is called depending on
        the correlation of all the individuals.
        -> Starts at alpha%.
        -> Increases to [-alpha to alpha] depending on the correlation of the individuals.
        :param alpha: The initial value of the decrement.
        :param likelihood: The likelihood of the individuals to be repeated.
        """
        self.base_alpha = tf.constant(alpha, dtype=tf.float32)
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32)
        self.likelihood = tf.Variable(initial_value=likelihood, dtype=tf.float32)

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        # Get the cross correlation:
        cross_correlation = tf.matmul(individuals, tf.transpose(individuals))
        # Maximum of the correlation:
        normalization_term = tf.math.reduce_max([tf.math.reduce_max(cross_correlation), 0.001])
        # Normalize the correlation:
        correlation = tf.math.divide(cross_correlation, normalization_term)
        # Get the new likelihood:
        new_likelihood = tf.abs(tf.math.reduce_mean(correlation))
        # Get the new alpha = alpha + alpha0 * ([0, 1] - likelihood):
        new_a = tf.clip_by_value(tf.math.add(self.alpha, tf.math.multiply(self.base_alpha,
                                                                          (new_likelihood -
                                                                           self.likelihood))),
                                 0, 1)
        self.alpha.assign(new_a)
        self.likelihood.assign(new_likelihood)

        # Get random mask to be added to the individuals with alpha probability:
        mask = tf.random.uniform(shape=tf.shape(individuals), minval=0, maxval=1) < self.alpha

        # Get the negative mutation:
        negative_mutation = tf.multiply(individuals,
                                        (1 + tf.abs(tf.random.normal(shape=tf.shape(individuals),
                                                                     mean=0, stddev=new_a))))
        # Get the new individuals:
        new_individuals = tf.where(mask, individuals, negative_mutation)
        return new_individuals
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
