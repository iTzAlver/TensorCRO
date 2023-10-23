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
class ParticleSwarmOptimization(CROSubstrate):
    def __init__(self, directives: tf.Tensor, inertia: float = 0.5, cognition: float = 1., social: float = 1.):
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
        self.register = None

    def _call(self, individuals: tf.Tensor, **kwargs):
        # Recover register values:
        if self.register is not None:
            # Old parameters:
            old_individuals = self.register[0]
            old_speeds = self.register[1]
            old_best_achieved = self.register[2]
            old_best_position = self.register[3]

            # See which individuals are inside old individuals:
            _ = tf.expand_dims(individuals, axis=1)
            occurrence_matrix = tf.reduce_all(tf.equal(_, old_individuals), axis=-1)


            # Get the index where the individuals are in the old register:
            index_in_register = tf.where(old_mark_in_register)

            # Get the speed of the last parameters:
            speeds = tf.where(mask_in_register, self.register[1], tf.zeros_like(individuals))
            best_achieved = tf.where(mask_in_register, self.register[2], individuals)
            best_position = tf.where(mask_in_register, self.register[3], 1)
        else:
            speeds = tf.zeros_like(individuals)
            best_achieved = individuals
            old_best_position = tf.ones((individuals.shape[0],))
        # Get the best particle of the current epoch:
        best_particle = individuals[0]

        # Inertia factor:
        v_inertia = self.inertia * speeds  # Done.
        # Cognition factor:
        v_congnition = self.cognition * (best_achieved - individuals)
        # Social factor:
        v_social = self.social * (best_particle - individuals)
        # Update the speed:
        speeds = v_inertia + v_social + v_congnition
        # Update the position:
        new_particles = individuals + speeds
        # Clip the values:
        new_particles_clipped = tf.clip_by_value(new_particles, self.directives[0], self.directives[1])
        # Get the new best position:
        this_position = tf.linspace(0., 1., individuals.shape[0])
        new_best_position = tf.where(this_position < old_best_position, this_position, old_best_position)
        # Get the new best achieved:
        new_best_achieved = tf.where(tf.expand_dims(this_position < old_best_position, axis=-1),
                                     individuals, best_achieved)
        # Save register values:
        self.register = (new_particles_clipped, speeds, new_best_achieved, new_best_position)
        return new_particles_clipped
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
