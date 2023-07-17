# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import time
import numpy as np


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class PSOAlgorithm:
    def __init__(self, number_particles, inertia=0.5, cognition=1., social=1.):
        """
        This function initializes the Particle Swarm Optimization algorithm.
        :param number_particles: The number of particles.
        :param inertia: The inertia factor.
        :param cognition: The cognition factor.
        :param social: The social factor.
            Velocity (v) of individual i in the population (pop) in parameter (p) in epoch (t):
                v[i][p][t + 1] =    inertia * v[i][p][t] +
                                    social_factor * (pop[i=best][p][t] - pop[i][p][t]) +
                                    cognition_factor * (pop[i][p][t=best] - pop[i][p][t])
        """
        self.particles = number_particles
        self.inertia = inertia
        self.cognition = cognition
        self.social = social
        self.register = None

    def __initialization(self, bounds):
        """
        This function initializes the population of the PSO algorithm.
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :return: A numpy array with the initial population.
        """
        if isinstance(self.particles, int):
            minimum = bounds[0]
            maximum = bounds[1]
            return np.random.uniform(minimum, maximum, size=(self.particles, len(minimum)))
        elif isinstance(self.particles, np.ndarray):
            return self.__clip(self.particles, bounds)

    def __mutation(self, particles, fitness, bounds):
        """
        This function performs the Gaussian mutation operation.
        :param particles: A numpy array with the children.
        :param fitness: A numpy array with the fitness values of the individuals.
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :return: A numpy array.
        """
        # Parameters:
        speeds = self.register[0]
        old_bests = self.register[1]
        old_positions_in_best = self.register[2]
        best_particle = particles[np.argmax(fitness)]
        # Update bests:
        mask = np.expand_dims(fitness > old_bests, axis=-1)
        new_bests = np.where(fitness > old_bests, fitness, old_bests)
        new_positions_in_best = np.where(mask, particles, old_positions_in_best)
        # Inertia factor:
        v_inertia = self.inertia * speeds  # Done.
        # Cognition factor:
        v_congnition = self.cognition * (new_positions_in_best - particles)
        # Social factor:
        v_social = self.social * (best_particle - particles)  # Done.
        # Update the speed:
        speeds = v_inertia + v_social + v_congnition
        # Update the register:
        self.register = (speeds, new_bests, new_positions_in_best)
        # Update the position:
        new_particles = particles + speeds
        # Clip the values:
        return self.__clip(new_particles, bounds)

    def fit(self, fitness_function, bounds, max_iterations=1000, seed=0, verbose=False, initialize=False,
            time_limit: int = None, evaluation_limit: int = None):
        """
        This function performs the genetic algorithm.
        :param fitness_function: A function that receives a numpy array with the individuals and returns a numpy array
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :param max_iterations: The maximum number of iterations.
        :param seed: The seed for the random number generator.
        :param initialize: A boolean or array to initialize the population.
        :param verbose: A boolean to print the results.
        :param time_limit: A float with the maximum time to run the algorithm.
        :param evaluation_limit: An integer with the maximum number of evaluations.
        :return: A numpy array with the best individuals.
        """
        # Set the seed and time:
        tik = time.perf_counter()
        np.random.seed(seed)
        # Initialize the population:
        if isinstance(initialize, np.ndarray):
            individuals = self.__clip(initialize, bounds)
        else:
            individuals = self.__initialization(bounds)
        # Evaluate the fitness of the individuals:
        fitness_values = -fitness_function(individuals)
        # Initialize the register:
        self.register = (np.zeros_like(individuals), fitness_values, individuals)
        # Main loop:
        for iterations in range(max_iterations):
            # Perform the mutation operation:
            individuals = self.__mutation(individuals, fitness_values, bounds)
            # Evaluate the fitness of the children:
            fitness_values = -fitness_function(individuals)
            # Print the results:
            if verbose:
                print(f"PS: Iteration: {iterations:3d} | Best fitness: {-fitness_values[0]:.3f} "
                      f"| Best individual: {individuals[0]}")
            # Check the time limit:
            if time_limit:
                if time.perf_counter() - tik > time_limit:
                    break
            # Check the evaluation limit:
            if evaluation_limit:
                if fitness_function.number_of_evaluations > evaluation_limit:
                    break
        # Return the best individuals:
        return individuals, -fitness_values

    @staticmethod
    def __clip(individuals, bounds):
        """
        This function clips the individuals to the bounds.
        :param individuals: A numpy array with the individuals.
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :return: A numpy array with the clipped individuals.
        """
        return np.clip(individuals, bounds[0], bounds[1])

    def __repr__(self):
        return "Particle Swarm Optimization Algorithm in numpy."
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
