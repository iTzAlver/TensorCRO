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
import time

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
                 reef_shape: tuple = (10, 10),
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

    def fit(self, fitness_function: (TensorFlowFunction, Callable), individual_directives: tf.Tensor, save: bool = True,
            max_iter: int = 100, device: str = '/GPU:0', seed: int = None, init=None, shards=None, monitor=False,
            time_limit: int = None, evaluation_limit: int = None, minimize: bool = True) \
            -> tuple[tf.Tensor, tf.Tensor]:
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
        :param save: boolean to tell if the algorithm should save the progress.
        :param time_limit: Time limit for the algorithm in seconds.
        :param evaluation_limit: Number of max evaluations for the algorithm.
        :param minimize: Boolean to tell if the algorithm should minimize or maximize the fitness function.
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
        # Time limit tik:
        if time_limit:
            tik = time.perf_counter()
        else:
            tik = 0
        # Minimize or maximize:
        if minimize:
            reverse = -1
        else:
            reverse = 1
        # Using the selected device:
        with tf.device(device):
            __progress_bar = tf.keras.utils.Progbar(max_iter // shards)
            __p = None
            for _ in range(max_iter // shards):
                rf = self._fit(_fitness_function, individual_directives, shards, rf, reverse=reverse)
                __progress_bar.update(_)
                reef, fitness = rf
                fitness *= reverse
                if save:
                    self.__save_replay(reef, fitness)
                if monitor:
                    if __p is not None:
                        __p.terminate()
                    __p = self.watch_replay()
                if time_limit:
                    if time.perf_counter() - tik > time_limit:
                        break
                if evaluation_limit:
                    if fitness_function.number_of_evaluations > evaluation_limit:
                        break
            sorted_reef = tf.gather(tf.reshape(reef, (-1, tf.shape(individual_directives)[-1])),
                                    tf.argsort(tf.reshape(fitness, (-1,)), direction='DESCENDING'))
            __progress_bar.update(max_iter // shards)
            sorted_fitness = tf.gather(tf.reshape(fitness, (-1,)), tf.argsort(tf.reshape(fitness, (-1,)),
                                                                              direction='DESCENDING'))
            return sorted_reef, sorted_fitness

    @tf.function
    def _fit(self, fitness_function: tf.function, individual_directives, max_iter: int = 100, init=None,
             reverse: int = 1) -> tuple:
        # Precompute some useful parameters:
        __n_parameters = tf.shape(individual_directives)[-1]
        __n_substrates = self._number_of_reefs
        __n_columns = self._substrate_segmentation
        __parameter_maxmin_diff = tf.subtract(individual_directives[1], individual_directives[0])  # Max - Min.

        # Reef tensors follow the shape (N_substrates, N_rows, N_columns / N_substrates, N_parameters).
        __reef_shape = tf.convert_to_tensor((__n_substrates, self._reef_shape[0], __n_columns, __n_parameters))
        __initial_reef_shape = tf.convert_to_tensor((self._reef_shape[0], self._reef_shape[1], __n_parameters))

        # Precompute Gaussian Deviation. (ORIGINAL CRO-SL: STILL USING GAUSSIAN MUTATION AS BROODING)
        # Gaussian recommended deviation by original SSS paper.
        parameter_deviations = tf.divide(__parameter_maxmin_diff, 100)

        # Create the numeric reef.
        if init is None:
            # Build random reef tensor
            reef = tf.random.uniform(__reef_shape, dtype=tf.float32, name='numeric_reef')
            reef = tf.add(tf.multiply(__parameter_maxmin_diff, reef), individual_directives[0], name='numeric_reef')
            reef = tf.clip_by_value(reef, individual_directives[0], individual_directives[1])  # Apply boundaries

            # Build fitness tensor
            occupied_idxs = tf.concat([tf.ones(self._n_init, dtype=tf.bool),
                                       tf.zeros(__initial_reef_shape[0] * __initial_reef_shape[1] - self._n_init,
                                                dtype=tf.bool)], axis=0)
            occupied_idxs = tf.reshape(tf.random.shuffle(occupied_idxs), __reef_shape[:-1])
            _occupied_cells = tf.boolean_mask(reef, occupied_idxs)
            _occupied_fitness = tf.cast(reverse * fitness_function(_occupied_cells), tf.float32)
            _initial_fitness = tf.ones(__reef_shape[:-1], dtype=tf.float32) * TF_INF
            fitness = tf.tensor_scatter_nd_update(_initial_fitness, tf.where(occupied_idxs), _occupied_fitness)
            fitness = tf.expand_dims(fitness, -1)

            #   [!] Version 1 - Evaluating all the reef - Deprecated:
            # ____to_fitness = tf.reshape(reef, (-1, __n_parameters))
            # __ = tf.where(__, tf.reshape(fitness_function(____to_fitness), __initial_reef_shape[:-1]), TF_INF)
            # fitness = tf.expand_dims(tf.reshape(__, __reef_shape[:-1]), -1)
        else:
            reef = tf.reshape(init[0], __reef_shape)
            fitness = tf.expand_dims(tf.reshape(init[1], __reef_shape[:-1]), -1)

        for _ in range(max_iter):
            # Separating corals into brooders and spawners
            n_corals_per_substrate = tf.reduce_sum(tf.where(tf.math.is_finite(fitness), 1, 0), axis=[1, 2, 3])
            n_spawners_per_substrate = tf.cast(tf.math.round(tf.multiply(self._fb,
                                                                         tf.cast(n_corals_per_substrate, tf.float32))),
                                               dtype=tf.int32)
            n_brooders_per_substrate = tf.reduce_sum(tf.subtract(n_corals_per_substrate, n_spawners_per_substrate))
            occupied_cells_idxs = tf.where(tf.math.is_finite(tf.squeeze(fitness)))
            occupied_cells_idxs_shuffled = tf.random.shuffle(occupied_cells_idxs)
            occupied_cells = tf.gather_nd(reef, occupied_cells_idxs_shuffled)
            _partitions = tf.dynamic_partition(occupied_cells, tf.cast(occupied_cells_idxs_shuffled[:, 0], tf.int32),
                                               __n_substrates)
            spawners = list()
            brooders = list()
            for _p, _ns in zip(_partitions, tf.dynamic_partition(n_spawners_per_substrate, tf.range(__n_substrates),
                                                                 __n_substrates)):
                spawners.append(_p[:_ns[0]])
                brooders.append(_p[_ns[0]:])

            # 1.- Broadcast spawning:
            larvae_spawners = self._substrates(self._substrate_functions, spawners)

            # 2.- Brooding:
            brooders = tf.concat(brooders, axis=0)
            mutation = tf.random.normal((n_brooders_per_substrate, __n_parameters), mean=0.0,
                                        stddev=parameter_deviations)
            larvae_brooders = tf.add(brooders, mutation)

            # 3.- Fragmentation:
            n_frags_per_substrate = tf.reduce_sum(tf.cast(tf.math.round(
                tf.multiply(self._fa, tf.cast(n_corals_per_substrate, tf.float32))), dtype=tf.int32))
            occupied_cells_idxs_shuffled = tf.random.shuffle(occupied_cells_idxs)
            occupied_cells = tf.gather_nd(reef, occupied_cells_idxs_shuffled)  # frag_params
            frag_fitness = tf.gather_nd(fitness, occupied_cells_idxs_shuffled[:n_frags_per_substrate])
            frag_larvae = occupied_cells[:n_frags_per_substrate]

            # 4.- Evaluation:
            larvae = tf.concat([larvae_spawners, larvae_brooders], axis=0)
            larvae = tf.clip_by_value(larvae, individual_directives[0], individual_directives[1])
            larvae_fitness = tf.expand_dims(tf.cast(reverse * fitness_function(larvae), tf.float32), -1)

            # 5. - First and second larvae setting:
            first_stage_larvae = larvae
            first_stage_larvae_fitness = larvae_fitness
            second_stage_larvae = frag_larvae
            second_stage_larvae_fitness = frag_fitness

            for larvae_params, larvae_fitness in [(first_stage_larvae, first_stage_larvae_fitness),
                                                  (second_stage_larvae, second_stage_larvae_fitness)]:
                for ___ in range(self._k):
                    attempted_cell_idxs = tf.random.shuffle(tf.reshape(tf.stack(
                        tf.meshgrid(tf.range(__reef_shape[0]), tf.range(__reef_shape[1]), tf.range(__reef_shape[2])),
                        axis=-1), (-1, 3)))[:tf.shape(larvae_params)[0]]
                    coral_fitness = tf.gather_nd(fitness, attempted_cell_idxs)
                    coral_params = tf.gather_nd(reef, attempted_cell_idxs)
                    larvae_wins = tf.greater(larvae_fitness, coral_fitness)

                    new_fitness = tf.where(larvae_wins, larvae_fitness, coral_fitness)
                    fitness = tf.tensor_scatter_nd_update(fitness, attempted_cell_idxs, new_fitness)
                    new_reef = tf.where(larvae_wins, larvae_params, coral_params)
                    reef = tf.tensor_scatter_nd_update(reef, attempted_cell_idxs, new_reef)
                    larvae_fitness = tf.where(larvae_wins, TF_INF, larvae_fitness)

            # 7.- Depredation:
            n_corals_per_substrate = tf.reduce_sum(tf.where(tf.math.is_finite(fitness), 1, 0), axis=[1, 2, 3])
            n_corals_to_depredate = tf.reduce_sum(tf.cast(tf.math.round(tf.multiply(self._fd,
                                                                                    tf.cast(n_corals_per_substrate,
                                                                                            dtype=tf.float32))),
                                                          dtype=tf.int32))

            occupied_cells_idxs_shuffled = tf.where(tf.math.is_finite(fitness))[:, :-1]
            occupied_cells_fitness = tf.squeeze(tf.gather_nd(fitness, occupied_cells_idxs_shuffled))
            depred_candidates_cell_idxs = tf.argsort(occupied_cells_fitness)[:n_corals_to_depredate]
            depred_candidates_fitness = tf.gather(occupied_cells_fitness, depred_candidates_cell_idxs)
            depred_candidates_idxs = tf.gather(occupied_cells_idxs_shuffled, depred_candidates_cell_idxs)

            _random_tensor = tf.random.uniform(tf.shape(depred_candidates_cell_idxs), minval=0, maxval=1)
            _was_predated_mask = tf.math.less(_random_tensor, self._pd)
            _new_fitness_list = tf.expand_dims(tf.where(_was_predated_mask, TF_INF, depred_candidates_fitness), -1)
            fitness = tf.tensor_scatter_nd_update(fitness, depred_candidates_idxs, _new_fitness_list)

        __return_value = tf.reshape(reef, __initial_reef_shape), tf.reshape(fitness, self._reef_shape)
        return __return_value

    @staticmethod
    def _substrates(substrates, spawners):
        """
        This function is used to perform the substrate crossover operation. It is called by the _crossover function.
        :param substrates: A list of functions that perform the substrate crossover operation.
        :param spawners: A list of reefs to perform the crossover operation on.
        :return: A tensor with the result of the crossover operation.
        """
        crossover = list()
        for substrate, spawner in zip(substrates, spawners):
            crossover.append(substrate(spawner))
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
    def watch_replay(path: str = __replay_path__, lock: bool = False, mp: bool = True) -> (None,
                                                                                           multiprocessing.Process):
        """
        This function is used to watch a replay in the GUI.
        :param path: A string with the path to the replay.
        :param lock: A boolean that indicates if the function should wait for the replay to finish.
        :param mp: A boolean that indicates if the function should run in a separate process.
        :return: A multiprocessing.Process object or None.
        """
        if mp:
            p = multiprocessing.Process(target=watch_replay, args=(path,))
            p.start()
            if lock:
                p.join()
            return p
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
