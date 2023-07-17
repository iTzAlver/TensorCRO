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
class SimulatedAnnealingAlgorithm:
    def __init__(self, mutation_variance=0.05):
        """
        This class implements the Simulated Annealing algorithm.
        """
        self.__mutation_variance = mutation_variance

    def __mutation(self, parent, bounds):
        """
        This function performs the mutation operation.
        :param parent: A numpy array with the parent.
        :return: A numpy array.
        """
        # Bound diff:
        bound_diff = bounds[1] - bounds[0]
        # Gaussian mutation:
        noise = bound_diff * np.random.normal(0, self.__mutation_variance, size=parent.shape)
        child = parent + noise
        # Clip the child:
        child = self.__clip(child, bounds)
        # Return the child:
        return child

    @staticmethod
    def __selection(individual, child, fitness_value, fitness_child, temperature):
        """
        This function performs the selection operation.
        :param individual: A numpy array with the parent.
        :param child: A numpy array with the child.
        :param fitness_value: The fitness of the parent.
        :param fitness_child: The fitness of the child.
        :param temperature: The temperature.
        :return:
        """
        # Select the best individual:
        if fitness_child[0] > fitness_value[0]:
            return child, fitness_child
        else:
            prob = np.exp((fitness_child[0] - fitness_value[0]) / temperature)
            if np.random.uniform(0, 1) < prob:
                return child, fitness_child
            else:
                return individual, fitness_value

    def fit(self, fitness_function, bounds, max_iterations=1000, seed=0, verbose=False, initialize=False,
            time_limit=None, evaluation_limit=None):
        """
        This function performs the genetic algorithm.
        :param fitness_function: A function that receives a numpy array with the individuals and returns a numpy array
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :param max_iterations: The maximum number of iterations.
        :param seed: The seed for the random number generator.
        :param initialize: A boolean or array to initialize the population.
        :param verbose: A boolean to print the results.
        :param time_limit: An integer with the maximum time, in seconds, to run the algorithm.
        :param evaluation_limit: An integer with the maximum number of evaluations.
        :return: A numpy array with the best individuals.
        """
        # Set the seed and time:
        tik = time.perf_counter()
        np.random.seed(seed)
        # Initialize the population:
        if isinstance(initialize, np.ndarray):
            individual = self.__clip(initialize, bounds)
        else:
            individual = np.random.uniform(bounds[0], bounds[1], size=(len(bounds[0])))
        # Evaluate the fitness of the individuals:
        fitness_value = -fitness_function(np.array([individual]))
        # Main loop:
        for iterations in range(max_iterations):
            # Compute the temperature:
            temperature = 1 - iterations / max_iterations
            # Perform the mutation operation:
            child = self.__mutation(individual, bounds)
            # Evaluate the fitness of the children:
            fitness_child = -fitness_function(np.array([child]))
            # Perform the selection operation:
            individual, fitness_value = self.__selection(individual, child, fitness_value, fitness_child, temperature)
            # Print the results:
            if verbose:
                print(f"SA: Iteration: {iterations:3d} | Best fitness: {-fitness_value[0]:.3f} "
                      f"| Best individual: {individual}")
            # Check the time limit:
            if time_limit:
                if time.perf_counter() - tik > time_limit:
                    break
            # Check the evaluation limit:
            if evaluation_limit:
                if fitness_function.number_of_evaluations > evaluation_limit:
                    break
        # Return the best individuals:
        return np.array([individual]), -fitness_value

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
        return "Simulated Annealing Algorithm in numpy."


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
