# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class LightCallback:
    def __init__(self, logger: logging.Logger = None, pile_callback=None, verbose=True):
        """
        Light callback class. This class is used to provide a callback to the fit method to see the optimization
        progress.
        :param logger: Logger to use. If None, the default logger is used.
        :param pile_callback: A function to be called each shard.
        :param verbose: If True, the callback will print information.
        """
        self.current_shard = 0
        self.best_fitness = -np.inf
        self.fitness_history = list()
        self.max_shards = None
        self.last_best_pop = None

        # Set up upper level callback.
        self.upper_level_callback = pile_callback
        self.verbose = verbose
        if logger is None:
            self.logging = logging
        else:
            self.logging = logger

    def exception_handler(self, exception):
        """
        This method is called when an exception is raised.
        :param exception: The exception raised.
        :return: True if the callback has been called correctly.
        """
        try:
            self.end(self.last_best_pop, where='./')
            self.logging.error(f'[LightCallback] Model saved, exception: {exception}')
            return True
        except Exception as e:
            self.logging.error(f'[LightCallback] Model not saved {e},\n\nException raised: {exception}')
            return False

    def end(self, best_solution: (np.ndarray, tf.Tensor) = None, where='./'):
        """
        This method is called when the fit method is finished.
        :param best_solution: Best solution found.
        :param where: Where to save the solution.
        :return: True if the callback has been called correctly.
        """
        # Create fitness plot.
        plt.plot(self.fitness_history, color='blue', linewidth=1.5, linestyle='solid', marker='o')
        plt.title(f'Fitness function ({self.current_shard} shards)')
        plt.xlabel('Shards')
        plt.xticks(np.arange(0, self.current_shard, min(self.current_shard, 100)))
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.savefig(where + '/fitness.png')
        plt.close()
        try:
            # Send message to slack.
            np.save(where + '/solution.npy', best_solution)
            self.logging.info(f'[LightCallback] Model saved, best fitness: {self.best_fitness}')
            return True
        except Exception as e:
            self.logging.error(f'[LightCallback] Error while ending the callback: {e}')
            return False

    def __call__(self, *args, **kwargs):
        """
        This method is called when the callback is called.
        :param args: Arguments of the callback. [0] is the population, [1] is the fitness [2] is the max shards.
        :param kwargs: Not used.
        :return: True if the callback has been called correctly.
        """
        # Collect information.
        sorted_fitness = args[1]
        max_shards = args[2]
        best_fitness = float(sorted_fitness[0])
        self.fitness_history.append(best_fitness)
        self.max_shards = max_shards

        # Update shard and best fitness.
        self.current_shard += 1
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.last_best_pop = args[0].numpy()

        # Print information.
        if self.verbose:
            self.logging.info(f'[LightCallback] Shard {self.current_shard}/{max_shards}, '
                              f'best fitness: {self.best_fitness:.2f}, ')
        # Call upper level callback.
        if self.upper_level_callback is not None:
            self.upper_level_callback(*args, **kwargs)
        return True
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
