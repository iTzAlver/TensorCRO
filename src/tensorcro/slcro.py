# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import multiprocessing
import shutil
import os
import json
import tensorflow as tf
import numpy as np
from typing import Callable
from tensorflow.python.eager.polymorphic_function.polymorphic_function import Function as TensorFlowFunction
from .__special__ import __replay_path__
from .replay import watch_replay

TF_INF = tf.constant(-np.inf, dtype=tf.float32)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class TensorCro:
    def __init__(self,
                 reef_shape: tuple[int, int] = (10, 10),
                 rho: float = .4,
                 fb: float = .80,
                 fd: float = .50,
                 pd: float = .20,
                 fa: float = .15,
                 k: int = 3,
                 subs: list = None):
        """
        This class implements the Coral Reef Optimization algorithm.
        It is a metaheuristic algorithm that uses the concept of coral reefs to optimize a function.
        It is a population based algorithm that uses a population of corals to find the best solution.
        Each coral is a solution to the problem and the algorithm will evolve the population to find the best solution.
        The algorithm is based on the idea of coral spawning and depredation.
        The spawning of corals is used to create new solutions and the depredation is used to remove solutions that are
        not good enough.
        The algorithm is also based on the idea of coral reefs being composed of different substrates.
        This means that the algorithm will evolve the population in different ways depending on the substrate where the
        coral is located. The algorithm will evolve the population in a similar way to the genetic algorithm.
        The algorithm will use the fitness function to evaluate the solutions and will evolve the population using
        the fitness function.

        :param reef_shape: Coral shape, telling the maximum number of individuals in the reef. As a tuple of two int.
        :param rho: Percentage of initial occupation of the reef, bounded between 0 and 1.
        :param fb: Ratio of current corals that act as spawners, bounded between 0 and 1.
        :param fa: Ratio of current corals that act as fragmentation reproduction, bounded between 0 and 1.
        :param fd: Depredation ratio.
        :param pd: Probability of depredation.
        :param k: Maximum attempts for larva setting.
        :param subs: Tells the substrates of the reef, divisible by the reef columns.
        """
        self._reef_shape = tf.constant(reef_shape, dtype=tf.int32)
        # Initialization parameters:
        self._n_init = tf.cast(tf.math.round(rho * reef_shape[0] * reef_shape[1]), dtype=tf.int32)
        # Broadcast spawning parameters:
        self._fb = tf.constant(fb, dtype=tf.float32)
        # Fragmentation parameters:
        self._fa = tf.constant(fa, dtype=tf.float32)
        # Larva parameters:
        self._k = tf.constant(k, dtype=tf.int32)
        # Depredation parameters:
        self._pd = tf.constant(pd, dtype=tf.float32)
        self._fd = tf.constant(fd, dtype=tf.float32)
        # Substrates:
        self._substrate_functions = subs
        if self._reef_shape[1] % len(subs) != 0:
            raise ValueError('TensorCRO: The reef columns are not divisible by the number of selected substrates.')
        else:
            self._substrate_segmentation = tf.constant(self._reef_shape[1] // len(subs), dtype=tf.int32)
            self._number_of_reefs = tf.constant(len(subs), dtype=tf.int32)
        # Save utils:
        self.shards = None
        self.seed = None
        self.n_fit = None

    def fit(self, fitness_function: (TensorFlowFunction, Callable), individual_directives: tf.Tensor,
            max_iter: int = 100, device: str = '/GPU:0', seed: int = None, init=None, shards=None, monitor=False) \
            -> tf.Tensor:
        """
        This function is the main loop of the algorithm. It will run the algorithm until the maximum number of
        iterations is reached or the fitness function returns a value that is considered optimal.
        :param fitness_function: function to be optimized
        :param individual_directives: Tensor with the minimum and maximum values for each individual. I.e:
        [(0.5, 0.6, 0.7), (0.8, 0.9, 1.0)] for 3 individuals (first minimum, then maximum).
        :param max_iter: maximum number of iterations.
        :param device: device to use for the calculations.
        :param seed: seed for the random number generator.
        :param init: initial population and fitness to use as a tuple.
        :param shards: number of shards to use for the computation.
        :param monitor: boolean to tell if the algorithm should track the progress.
        :return: the best solution found.
        """
        # Formatting directives:
        if not isinstance(individual_directives, tf.Tensor):
            individual_directives = tf.convert_to_tensor(individual_directives, dtype=tf.float32)
        # Clearing save buffer:
        if os.path.exists(__replay_path__):
            shutil.rmtree(__replay_path__)
        os.mkdir(__replay_path__)
        # Set up the seed:
        if seed is None:
            seed = tf.random.uniform((1,), minval=0, maxval=2147483647, dtype=tf.int32)
        self.seed = int(seed)
        tf.random.set_seed(self.seed)
        # Set up shards and initial population:
        if shards is None:
            shards = max_iter
        self.shards = shards
        rf = init
        self.n_fit = 0
        # Checkout fitness.
        if not isinstance(fitness_function, TensorFlowFunction):
            _fitness_function = _function_cast_(fitness_function)
        else:
            _fitness_function = fitness_function
        # Using the selected device:
        with tf.device(device):
            for _ in range(max_iter // shards):
                rf = self._fit(_fitness_function, individual_directives, shards, rf)
                reef, fitness = rf
                self.__save_replay(reef, fitness)
                if monitor:
                    self.watch_replay()
            sorted_reef = tf.gather(tf.reshape(reef, (-1, tf.shape(individual_directives)[-1])),
                                    tf.argsort(tf.reshape(fitness, (-1,)), direction='DESCENDING'))
            return sorted_reef

    @tf.function
    def _fit(self, fitness_function: tf.function, individual_directives, max_iter: int = 100, init=None) \
            -> tuple[tf.Tensor, tf.Tensor]:
        # Precompute some useful parameters:
        __number_of_parameters = tf.shape(individual_directives)[-1]  # Number of parameters for each coral.
        __parameter_diff = tf.subtract(individual_directives[1], individual_directives[0])  # Max - Min.
        __number_of_reefs = self._number_of_reefs
        __number_of_columns = self._substrate_segmentation
        __reef_shape = tf.convert_to_tensor((__number_of_reefs, self._reef_shape[0],  # Reefs-Rows-Columns-Params.
                                             __number_of_columns, __number_of_parameters))
        __original_reef_shape = tf.convert_to_tensor((self._reef_shape[0], self._reef_shape[1],
                                                      __number_of_parameters))  # R-C-P.
        # Precompute Gaussian Deviation. (ORIGINAL CRO-SL: STILL USING GAUSSIAN MUTATION AS BROODING)
        deviations = tf.divide(__parameter_diff, 100)  # Gaussian recommended deviation by original SSS paper.
        # Create the numeric reef.
        if init is None:
            # Build reef.
            reef = tf.random.uniform(__reef_shape, dtype=tf.float32, name='numeric_reef')  # Random reef and scale.
            reef = tf.add(tf.multiply(__parameter_diff, reef), individual_directives[0], name='numeric_reef')
            reef = tf.clip_by_value(reef, individual_directives[0], individual_directives[1])  # Apply boundaries.
            # Create the fitness storage.
            __ = tf.concat([tf.ones(self._n_init, dtype=tf.bool),
                            tf.zeros(__original_reef_shape[0] * __original_reef_shape[1] - self._n_init,
                                     dtype=tf.bool)], axis=0)
            __ = tf.reshape(tf.random.shuffle(__), __reef_shape[:-1])
            _selected_grid = tf.boolean_mask(reef, __)
            _selected_fitness = tf.cast(fitness_function(_selected_grid), tf.float32)
            _full_fitness = tf.ones(__reef_shape[:-1], dtype=tf.float32) * TF_INF
            fitness = tf.tensor_scatter_nd_update(_full_fitness, tf.where(__), _selected_fitness)
            fitness = tf.expand_dims(fitness, -1)
            #   [!] Version 1 - Evaluating all the reef - Deprecated:
            # ____to_fitness = tf.reshape(reef, (-1, __number_of_parameters))
            # __ = tf.where(__, tf.reshape(fitness_function(____to_fitness), __original_reef_shape[:-1]), TF_INF)
            # fitness = tf.expand_dims(tf.reshape(__, __reef_shape[:-1]), -1)
        else:
            reef = tf.reshape(init[0], __reef_shape)
            fitness = tf.expand_dims(tf.reshape(init[1], __reef_shape[:-1]), -1)
        number_alive_corals = tf.reduce_sum(tf.where(tf.math.is_finite(fitness), 1, 0), axis=[1, 2, 3])
        __progress_bar = tf.keras.utils.Progbar(max_iter)
        for _ in range(max_iter):
            # Some initial computation:
            __progress_bar.update(_)
            number_spawners = tf.cast(tf.math.round(tf.multiply(self._fb, tf.cast(number_alive_corals,
                                                                                  tf.float32))), dtype=tf.int32)
            number_brooders = tf.reduce_sum(tf.subtract(number_alive_corals, number_spawners))
            number_fragmentators = tf.reduce_sum(tf.cast(tf.math.round(
                tf.multiply(self._fa, tf.cast(number_alive_corals, tf.float32))), dtype=tf.int32))
            alive_positions = tf.where(tf.math.is_finite(tf.squeeze(fitness)))
            alive_positions_shuffled = tf.random.shuffle(alive_positions)
            alive_positions_shuffled_frag = tf.random.shuffle(alive_positions_shuffled)
            alive_corals = tf.gather_nd(reef, alive_positions_shuffled)
            alive_corals_frag = tf.gather_nd(reef, alive_positions_shuffled_frag)
            # Shuffle the corals and split them into spawners, brooders and fragmentators:
            _partitions = tf.dynamic_partition(alive_corals, tf.cast(alive_positions_shuffled[:, 0], tf.int32),
                                               __number_of_reefs)
            spawners = list()
            brooders = list()
            for _p, _ns in zip(_partitions, tf.dynamic_partition(number_spawners, tf.range(__number_of_reefs),
                                                                 __number_of_reefs)):
                spawners.append(_p[:_ns[0]])
                brooders.append(_p[_ns[0]:])

            # 1.- Broadcast spawning:
            larvae_spawners = self._substrates(self._substrate_functions, spawners)

            # 2.- Brooding:
            brooders = tf.concat(brooders, axis=0)
            mutation = tf.random.normal((number_brooders, __number_of_parameters), mean=0.0, stddev=deviations)
            larvae_brooders = tf.add(brooders, mutation)

            # 3.- Fragmentation:
            larvae_fragmentators = alive_corals_frag[:number_fragmentators]

            # 4.- Evaluation:
            larvae = tf.concat([larvae_spawners, larvae_brooders, larvae_fragmentators], axis=0)
            larvae = tf.clip_by_value(larvae, individual_directives[0], individual_directives[1])
            larvae_fitness = tf.expand_dims(tf.cast(fitness_function(larvae), tf.float32), -1)
            first_stage_larvae = larvae[:-number_fragmentators]
            first_stage_larvae_fitness = larvae_fitness[:tf.shape(first_stage_larvae)[0]]
            second_stage_larvae = larvae[-number_fragmentators:]
            second_stage_larvae_fitness = larvae_fitness[-number_fragmentators:]

            # 5. - First larvae setting:
            for ___ in range(self._k):
                positions_larvae = tf.random.shuffle(tf.reshape(tf.stack(
                    tf.meshgrid(tf.range(__reef_shape[0]), tf.range(__reef_shape[1]), tf.range(__reef_shape[2])),
                    axis=-1), (-1, 3)))[:tf.shape(first_stage_larvae)[0]]
                positioned_fitness = tf.gather_nd(fitness, positions_larvae)
                positioned_reef = tf.gather_nd(reef, positions_larvae)
                larvae_wins = tf.greater(first_stage_larvae_fitness, positioned_fitness)
                new_fitness = tf.where(larvae_wins, first_stage_larvae_fitness, positioned_fitness)
                fitness = tf.tensor_scatter_nd_update(fitness, positions_larvae, new_fitness)
                new_reef = tf.where(larvae_wins, first_stage_larvae, positioned_reef)
                reef = tf.tensor_scatter_nd_update(reef, positions_larvae, new_reef)
                first_stage_larvae_fitness = tf.where(larvae_wins, TF_INF, first_stage_larvae_fitness)
            # 6.- Second larvae setting:
            for ___ in range(self._k):
                positions_larvae = tf.random.shuffle(tf.reshape(tf.stack(
                    tf.meshgrid(tf.range(__reef_shape[0]), tf.range(__reef_shape[1]), tf.range(__reef_shape[2])),
                    axis=-1), (-1, 3)))[:tf.shape(second_stage_larvae)[0]]
                positioned_fitness = tf.gather_nd(fitness, positions_larvae)
                positioned_reef = tf.gather_nd(reef, positions_larvae)
                larvae_wins = tf.greater(second_stage_larvae_fitness, positioned_fitness)
                new_fitness = tf.where(larvae_wins, second_stage_larvae_fitness, positioned_fitness)
                fitness = tf.tensor_scatter_nd_update(fitness, positions_larvae, new_fitness)
                new_reef = tf.where(larvae_wins, second_stage_larvae, positioned_reef)
                reef = tf.tensor_scatter_nd_update(reef, positions_larvae, new_reef)
                second_stage_larvae_fitness = tf.where(larvae_wins, TF_INF, second_stage_larvae_fitness)
            # 7.- Depredation:
            number_alive_corals = tf.reduce_sum(tf.where(tf.math.is_finite(fitness), 1, 0), axis=[1, 2, 3])
            number_depredated_corals = tf.reduce_sum(tf.cast(tf.math.round(tf.multiply(self._fd,
                                                                                       tf.cast(number_alive_corals,
                                                                                               dtype=tf.float32))),
                                                             dtype=tf.int32))
            predation_positions = tf.where(tf.math.is_finite(fitness))[:, :-1]
            predation_values = tf.squeeze(tf.gather_nd(fitness, predation_positions))
            predation_indices = tf.argsort(predation_values)[:number_depredated_corals]
            depredation_values = tf.gather(predation_values, predation_indices)
            depredation_positions = tf.gather(predation_positions, predation_indices)

            _random_tensor = tf.random.uniform(tf.shape(predation_indices), minval=0, maxval=1)
            _predation_bool = tf.math.less(_random_tensor, self._pd)
            inf_or_not = tf.expand_dims(tf.where(_predation_bool, TF_INF, depredation_values), -1)
            fitness = tf.tensor_scatter_nd_update(fitness, depredation_positions, inf_or_not)
            number_alive_corals = tf.reduce_sum(tf.where(tf.math.is_finite(fitness), 1, 0), axis=[1, 2, 3])
        __progress_bar.update(max_iter)
        return tf.reshape(reef, __original_reef_shape), tf.reshape(fitness, self._reef_shape)

    @staticmethod
    def _substrates(substrates, reefs):
        """
        This function is used to perform the substrate crossover operation. It is called by the _crossover function.
        :param substrates: A list of functions that perform the substrate crossover operation.
        :param reefs: A list of reefs to perform the crossover operation on.
        :return: A tensor with the result of the crossover operation.
        """
        crossover = list()
        for substrate, reef in zip(substrates, reefs):
            crossover.append(substrate(reef))
        return tf.concat(crossover, axis=0)

    def __save_replay(self, reef, fitness):
        """
        This function is used to save a tensor to a file.
        :param fitness: A tensor with the fitness values.
        :param reef: A tensor with the reef values.
        :return: None
        """
        self.n_fit += 1
        tensor_np = fitness.numpy()
        tensor2_np = reef.numpy()
        np.save(f'{__replay_path__}/fitness_{self.n_fit}.npy', tensor_np)
        np.save(f'{__replay_path__}/reef_{self.n_fit}.npy', tensor2_np)
        # Save the name of the subs in a json file:
        sbs = [sub.__repr__() for sub in self._substrate_functions]
        with open(f'{__replay_path__}/config.json', 'w') as f:
            json.dump({'names': sbs, 'shards': self.shards, 'seed': self.seed}, f, indent=2)

    @staticmethod
    def save_replay(path):
        """
        This function is used to save replay to a directory.
        :param path: A string with the path to save the replay.
        :return: None
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.copytree(__replay_path__, path)

    @staticmethod
    def watch_replay(path: str = __replay_path__, lock: bool = False, mp: bool = True):
        """
        This function is used to watch a replay in the GUI.
        :param path: A string with the path to the replay.
        :param lock: A boolean that indicates if the function should wait for the replay to finish.
        :param mp: A boolean that indicates if the function should run in a separate process.
        :return: A multiprocessing.Process object.
        """
        if mp:
            p = multiprocessing.Process(target=watch_replay, args=(path,))
            p.start()
            if lock:
                p.join()
        else:
            watch_replay(path)

    def __repr__(self):
        return "<TensorCRO instance initialized>"


def _function_cast_(function):
    def casted_python_function(inputs):
        return tf.py_function(function, inp=[inputs], Tout=tf.float32)
    return casted_python_function
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
