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
class GeneticAlgorithm:
    def __init__(self, individuals, mutation_rate, crossover_rate, mutation_variance):
        self.individuals = individuals
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_variance = mutation_variance

    def __initialization(self, bounds):
        """
        This function initializes the population of the genetic algorithm.
        :param bounds: A list of tuples with the minimum and maximum values for each dimension.
        :return: A numpy array with the initial population.
        """
        if isinstance(self.individuals, int):
            minimum = bounds[0]
            maximum = bounds[1]
            return np.random.uniform(minimum, maximum, size=(self.individuals, len(minimum)))
        elif isinstance(self.individuals, np.ndarray):
            return self.__clip(self.individuals, bounds)

    def __crossover(self, individuals):
        """
        This function performs the crossover operation.
        :param individuals: A numpy array with the individuals.
        :return: A numpy array with the crossovered individuals.
        """
        rate = self.crossover_rate
        _individuals = individuals.copy()
        # Shuffle individuals:
        np.random.shuffle(_individuals)
        # Select the individuals to crossover:
        fathers = _individuals[np.random.choice(len(_individuals), size=int(len(_individuals) * rate), replace=False)]
        # Select the individuals to crossover with:
        mothers = _individuals[np.random.choice(len(_individuals), size=int(len(_individuals) * rate),
                                                replace=False)]
        # Create a gene mask to select father or mother:
        gene_mask = np.random.choice([True, False], size=fathers.shape, p=[0.5, 0.5])
        # Create the children:
        children = np.where(gene_mask, fathers, mothers)
        return children

    def __mutation(self, children, bounds):
        """
        This function performs the Gaussian mutation operation.
        :param children: A numpy array with the children.
        :return: A numpy array.
        """
        rate = self.mutation_rate
        variance = self.mutation_variance
        bound_diff = np.array(bounds[1]) - np.array(bounds[0])
        # Select the genes to mutate:
        gene_mask = np.random.choice([True, False], size=children.shape, p=[rate, 1-rate])
        # Create a gene mask to select father or mother:
        mutation = bound_diff * np.random.normal(0, variance, size=children.shape)
        # Create the children:
        children = np.where(gene_mask, children + mutation, children)
        # Return the children NOT clipped to the bounds:
        return children

    @staticmethod
    def __selection(individuals, children, fitness_values, fitness_children):
        """
        This function performs the selection operation.
        :param individuals: A numpy array with the individuals.
        :param children: A numpy array with the children.
        :param fitness_values: A numpy array with the fitness values.
        :param fitness_children: A numpy array with the fitness values of the children.
        :return: A numpy array with the selected individuals.
        """
        # Sort the individuals and filter the worst ones from max to min:
        sorted_individuals = individuals[np.argsort(fitness_values)][::-1][:-len(children)]
        sorted_fitness = fitness_values[np.argsort(fitness_values)][::-1][:-len(children)]
        # Concat the children and the individuals:
        new_individuals = np.concatenate((sorted_individuals, children))
        new_fitness = np.concatenate((sorted_fitness, fitness_children))
        # Sort the individuals and fitness:
        new_sorted_individuals = new_individuals[np.argsort(new_fitness)][::-1]
        new_sorted_fitness = new_fitness[np.argsort(new_fitness)][::-1]
        # Return the selected individuals:
        return new_sorted_individuals, new_sorted_fitness

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
            self.individuals = self.__clip(initialize, bounds)
        individuals = self.__initialization(bounds)
        # Evaluate the fitness of the individuals:
        fitness_values = -fitness_function(individuals)
        # Main loop:
        for iterations in range(max_iterations):
            # Perform the crossover operation:
            children = self.__crossover(individuals)
            # Perform the mutation operation:
            children = self.__mutation(children, bounds)
            # Evaluate the fitness of the children:
            fitness_children = -fitness_function(children)
            # Perform the selection operation:
            individuals, fitness_values = self.__selection(individuals, children, fitness_values, fitness_children)
            # Print the results:
            if verbose:
                print(f"GA: Iteration: {iterations:3d} | Best fitness: {-fitness_values[0]:.3f} "
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
        return "Genetic Algorithm in numpy."
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
