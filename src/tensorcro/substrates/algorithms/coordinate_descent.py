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
class CoordinateDescentSubstrate(CROSubstrate):
    def __init__(self, bounds: tf.Tensor, epsilon: float = 1.):
        """
        This substrate is used to perform a coordinate descent. It searches for the best individual in each
        dimension, split by epsilon.
        :param bounds: The bounds of the individuals.
        :param epsilon: The width of the coordinate descent.
        """
        self.bounds = tf.constant(bounds, dtype=tf.float32)
        self.width = tf.constant(epsilon, dtype=tf.float32)
        self.pointer = tf.Variable(initial_value=0, dtype=tf.int32)

    def _call(self, population: tf.Tensor, **kwargs) -> tf.Tensor:
        # Get the size of the population:
        current_point = population[0]
        # Create new individuals from each direction of the current point:
        ranges = tf.range(self.bounds[0, self.pointer], self.bounds[1, self.pointer] + 1, self.width)
        # Tile individuals:
        individuals = tf.tile([current_point], (tf.shape(ranges)[0], 1))
        # Put zeros in the current pointer index:
        individuals = tf.concat([individuals[:, :self.pointer],
                                 ranges[:, tf.newaxis],
                                 individuals[:, self.pointer + 1:]], axis=-1)
        # Update the pointer:
        self.pointer.assign_add(1)
        # Put pointer to zero module the number of dimensions:
        self.pointer.assign(tf.math.mod(self.pointer, tf.shape(self.bounds)[-1]))
        # Clip the individuals:
        new_individuals = tf.clip_by_value(individuals, self.bounds[0], self.bounds[1])
        # Return the new individuals:
        return new_individuals
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
