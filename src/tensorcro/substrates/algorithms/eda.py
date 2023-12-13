# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
import tensorflow_probability as tfp
from ..substrate import CROSubstrate


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class EstimationDistributionAlgorithm(CROSubstrate):
    def __init__(self, bounds: tf.Tensor, init_prob: float = 0.65, lr: float = 0.1, proportion: float = 0.5,
                 top_select: int = 20, dist=tfp.distributions.Binomial):
        """
        This substrate is used to create a binomial distribution of the individuals.
        :param bounds: The bounds of the individuals.
        :param init_prob: The initial probability of the binomial distribution.
        :param lr: The learning rate of the binomial distribution.
        :param proportion: The proportion of the population to be used to estimate the parameters.
        :param top_select: The number of individuals to be used to estimate the parameters.
        :param dist: The distribution to be used. Binomial by default.
        """
        self.bounds = tf.constant(bounds, dtype=tf.float32)
        self.lr = tf.constant(lr, dtype=tf.float32)
        self.omlr = tf.constant(1 - lr, dtype=tf.float32)
        self.p = tf.Variable(initial_value=init_prob * tf.ones(bounds.shape[-1]), dtype=tf.float32)
        self.proportion = tf.constant(proportion, dtype=tf.float32)
        self.top_select = tf.constant(top_select, dtype=tf.int32)
        self.dist = dist

    def _call(self, population: tf.Tensor, **kwargs) -> tf.Tensor:
        # Select the first 20:;
        individuals = population[:self.top_select]
        # Estimate the parameters of the binomial distribution:
        __ = (tf.cast(tf.shape(individuals)[0], tf.float32) * (self.bounds[1] - self.bounds[0] + 1))
        p_hat = tf.math.reduce_sum(individuals, axis=0) / __
        # Apply lr to the parameters:
        p = tf.math.add(tf.math.multiply(self.p, self.omlr), tf.math.multiply(p_hat, self.lr))
        # Get the new individuals:
        new_individuals_unnclipped = self.dist(total_count=self.bounds[-1][-1], probs=p).\
            sample(tf.round(self.proportion * individuals.shape[0]))
        # Clip the individuals:
        new_individuals = tf.clip_by_value(new_individuals_unnclipped, self.bounds[0], self.bounds[1])
        # Return the new individuals:
        return new_individuals
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
