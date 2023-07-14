# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class FireflyAlgorithm:
    def __init__(self, particles, absorption_coefficient=0.2, mutation_variance=0.2):
        """
        This function initializes the FireFly algorithm.
        :param particles: The number of particles.
        :param absorption_coefficient: The absorption coefficient.
        """
        self.particles = particles
        self.absorption_coefficient = absorption_coefficient
        self.mutation_variance = mutation_variance

    def __initialization(self, bounds):
        """
        This function initializes the population of the FireFly algorithm.
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :return: A numpy array with the initial population.
        """
        if isinstance(self.particles, int):
            minimum = bounds[0]
            maximum = bounds[1]
            return np.random.uniform(minimum, maximum, size=(self.particles, len(minimum)))
        elif isinstance(self.particles, np.ndarray):
            return self.__clip(self.particles, bounds)

    def __mutation(self, parents, fitness, bounds):
        """
        This function performs the attraction mutation.
        :param parents: A numpy array with the children.
        :param fitness: A numpy array
        :return: A numpy array.
        """
        new_parents = parents.copy()
        greater_matrix = np.greater(fitness, fitness[:, np.newaxis])
        for nrow, row in enumerate(greater_matrix):
            for ncol, col in enumerate(row):
                if col:
                    new_parents[nrow] = self.__update_individual(parents[nrow], parents[ncol])
        return self.__clip(new_parents, bounds)

    def __update_individual(self, individual_0, individual_1):
        new_individual = (individual_1 - individual_0) * np.exp(-self.absorption_coefficient *
                                                                np.linalg.norm(individual_1 - individual_0))
        new_individual += individual_0 + self.mutation_variance * np.random.randn(len(individual_0))
        return new_individual

    def fit(self, fitness_function, bounds, max_iterations=1000, seed=0, verbose=False, initialize=False):
        """
        This function performs the genetic algorithm.
        :param fitness_function: A function that receives a numpy array with the individuals and returns a numpy array
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :param max_iterations: The maximum number of iterations.
        :param seed: The seed for the random number generator.
        :param initialize: A boolean or array to initialize the population.
        :param verbose: A boolean to print the results.
        :return: A numpy array with the best individuals.
        """
        # Set the seed:
        np.random.seed(seed)
        # Initialize the population:
        if isinstance(initialize, np.ndarray):
            individuals = self.__clip(initialize, bounds)
        else:
            individuals = self.__initialization(bounds)
        # Evaluate the fitness of the individuals:
        fitness_values = -fitness_function(individuals)
        # Main loop:
        for iterations in range(max_iterations):
            # Perform the mutation operation:
            individuals = self.__mutation(individuals, fitness_values, bounds)
            # Evaluate the fitness of the children:
            fitness_values = -fitness_function(individuals)
            # Sort the individuals:
            sorted_indexes = np.argsort(-fitness_values)
            individuals = individuals[sorted_indexes]
            fitness_values = fitness_values[sorted_indexes]
            # Print the results:
            if verbose:
                print(f"FF: Iteration: {iterations:3d} | Best fitness: {-fitness_values[0]:.3f} "
                      f"| Best individual: {individuals[0]}")
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
        return "FireFly Algorithm in numpy."
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
