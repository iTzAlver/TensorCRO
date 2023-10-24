# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from ..substrate import CROSubstrate
from ...slcro import TF_INF


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SimulatedAnnealing(CROSubstrate):
    def __init__(self, directives: tf.Tensor, kmax: int = 100, bandwidth: float = 0.2, shape: tuple = None):
        """
        This function initializes the Simulated Annealing algorithm.
        :param directives: Parameter specifications.
        :param kmax: The maximum number of iterations.
        :param bandwidth: The bandwidth of the mutation.
        :param shape: The shape of the substrate.
        """
        if not isinstance(directives, tf.Tensor):
            directives = tf.convert_to_tensor(directives)
        self.kmax = tf.constant(kmax, dtype=tf.float32)
        self.bandwidth = tf.constant(bandwidth)
        self.directives = directives
        # Annotations:
        r_shape = (shape[0], shape[1])
        self.iterations = tf.Variable(shape=r_shape, initial_value=tf.ones(r_shape), trainable=False)
        self.last_fitness_records = tf.Variable(shape=r_shape, initial_value=tf.ones(r_shape) * TF_INF, trainable=False)

    def _call(self, individuals: tf.Tensor, **kwargs):
        # Recover register values:
        new_ids = kwargs['ids']
        last_fitness = tf.gather_nd(self.last_fitness_records, new_ids)
        new_fitness = kwargs['fitness'][:, 0]
        iterations = tf.gather_nd(self.iterations, new_ids)
        # Check where the new fitness is better than the last one:
        better = tf.expand_dims(new_fitness > last_fitness, axis=-1)
        # Compute temperature T(k):
        tempreatures = 1. - (iterations / self.kmax)
        random_numbers = tf.random.uniform(shape=tf.shape(tempreatures), minval=0., maxval=1.)
        # Check where the random number is less than the temperature:
        less_than_temp = tf.expand_dims(random_numbers < tempreatures, axis=-1)
        # Or between better and less than temp:
        mask = tf.logical_or(better, less_than_temp)
        # Generate new individuals:
        mutation_individuals = individuals + tf.random.normal(shape=tf.shape(individuals), mean=0.,
                                                              stddev=self.bandwidth)
        # Clip the new individuals:
        mutation_individuals = tf.clip_by_value(mutation_individuals, self.directives[0], self.directives[1])
        new_individuals = tf.where(mask, mutation_individuals, individuals)
        # Save the new iterations:
        new_last_fitness_records = tf.tensor_scatter_nd_update(self.last_fitness_records, new_ids, new_fitness)
        new_iterations = tf.tensor_scatter_nd_update(self.iterations, new_ids, iterations + 1)
        self.last_fitness_records.assign(new_last_fitness_records)
        self.iterations.assign(new_iterations)
        # Return the new particles:
        return new_individuals
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
