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
class ParticleSwarmOptimization(CROSubstrate):
    def __init__(self, directives: tf.Tensor, inertia: float = 0.5, cognition: float = 1., social: float = 1.,
                 shape: tuple = None):
        """
        This function initializes the Particle Swarm Optimization algorithm.
        :param directives: Parameter specifications.
        :param inertia: The inertia factor.
        :param cognition: The cognition factor.
        :param social: The social factor.
            Velocity (v) of individual i in the population (pop) in parameter (p) in epoch (t):
                v[i][p][t + 1] =    inertia * v[i][p][t] +
                                    social_factor * (pop[i=best][p][t] - pop[i][p][t]) +
                                    cognition_factor * (pop[i][p][t=best] - pop[i][p][t])
        """
        if not isinstance(directives, tf.Tensor):
            directives = tf.convert_to_tensor(directives)
        self.inertia = tf.constant(inertia)
        self.cognition = tf.constant(cognition)
        self.social = tf.constant(social)
        self.directives = directives
        # Annotations:
        r_shape = (shape[0], shape[1], directives.shape[1])
        self.speeds = tf.Variable(shape=r_shape, initial_value=tf.zeros(r_shape), trainable=False)
        self.best_achieved = tf.Variable(shape=r_shape, initial_value=tf.zeros(r_shape), trainable=False)
        self.best_fitness = tf.Variable(shape=r_shape[:-1],
                                        initial_value=tf.ones(r_shape[:-1]) * TF_INF, trainable=False)

    def _call(self, individuals: tf.Tensor, **kwargs):
        # Recover register values:
        new_ids = kwargs['ids']
        speeds = tf.gather_nd(self.speeds, new_ids)
        best_achieved = tf.gather_nd(self.best_achieved, new_ids)
        best_fitness = tf.gather_nd(self.best_fitness, new_ids)
        # Get the best particle of the current epoch:
        best_particle = individuals[0]
        new_fitness = kwargs['fitness'][:, 0]
        # Get the new best achieved:
        new_best_achieved = tf.where(tf.expand_dims(new_fitness > best_fitness, axis=-1), individuals, best_achieved)
        new_best_fitness = tf.where(new_fitness > best_fitness, new_fitness, best_fitness)
        # Inertia factor:
        v_inertia = self.inertia * speeds
        # Cognition factor:
        v_congnition = self.cognition * (new_best_achieved - individuals)
        # Social factor:
        v_social = self.social * (best_particle - individuals)
        # Update the speed:
        speeds = v_inertia + v_social + v_congnition
        # Update the position:
        new_particles = individuals + speeds
        # Clip the values:
        new_particles_clipped = tf.clip_by_value(new_particles, self.directives[0], self.directives[1])
        # Save register values:
        all_new_speeds = tf.tensor_scatter_nd_update(self.speeds, new_ids, speeds)
        all_new_best_achieved = tf.tensor_scatter_nd_update(self.best_achieved, new_ids, new_best_achieved)
        all_new_best_fitness = tf.tensor_scatter_nd_update(self.best_fitness, new_ids, new_best_fitness)
        self.speeds.assign(all_new_speeds)
        self.best_achieved.assign(all_new_best_achieved)
        self.best_fitness.assign(all_new_best_fitness)
        # Return the new particles:
        return new_particles_clipped
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
