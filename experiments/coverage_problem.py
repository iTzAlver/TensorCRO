# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorcro import TensorCro, UniformCrossover, MultipointCrossover, HarmonySearch, \
    RandomSearch, ComposedSubstrate, Mutation
from coverage_problem import csv_to_numpy, format_array, conv2d
MAP_PATH = "./coverage_problem/points.csv"
DEVICE_PATH = "./coverage_problem/gen5.csv"


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Fitness:
    def __init__(self, coordinates, is_position, coverage_boolean, distance, cost, framework='numpy'):
        """
        This class calculates the fitness of each individual.
        :param coordinates: Array of coordinates.
        :param is_position:
        :param coverage_boolean:
        :param distance:
        :param cost:
        """
        # Set up main variables:
        self._coordinates = coordinates
        self._is_position = is_position
        self._coverage_boolean = coverage_boolean
        self.distance = distance
        self.cost = cost
        # Compute the zone matrix:
        max_x = coordinates[:, 0].max()
        max_y = coordinates[:, 1].max()
        # Get divisions:
        unique_x = np.unique(coordinates[:, 0])
        unique_y = np.unique(coordinates[:, 1])
        delta_x = min(np.abs(unique_x - np.roll(unique_x, 1)))
        delta_y = min(np.abs(unique_y - np.roll(unique_y, 1)))
        # Get number of divisions:
        num_x = round((max_x - min(unique_x)) / delta_x) + 1
        num_y = round((max_y - min(unique_y)) / delta_y) + 1
        # Get zone matrix:
        zone_matrix = np.zeros((num_x, num_y), dtype=np.int32)
        coordinates_index_x = np.array(coordinates[:, 0] / delta_x, dtype=np.int32)
        coordinates_index_y = np.array(coordinates[:, 1] / delta_y, dtype=np.int32)
        coordinates_index = np.array([coordinates_index_x, coordinates_index_y]).T
        # Cast '1' to zone matrix where coordinates are:
        zone_matrix[coordinates_index[:, 0], coordinates_index[:, 1]] = 1
        self.z = zone_matrix
        # Get coordinates where devices are:
        device_matrix = np.zeros((num_x, num_y), dtype=np.int32)
        device_coordinates = coordinates_index[is_position == 1]
        device_matrix[device_coordinates[:, 0], device_coordinates[:, 1]] = 1
        self.base_d = device_matrix
        __device_coordinates = [np.concatenate([np.expand_dims([_] * len(device_coordinates), 1),
                                                device_coordinates], axis=1)
                                for _, __ in enumerate(distance)]
        self.device_coordinates = np.concatenate(__device_coordinates, axis=0)
        # Kernel:
        self.k = np.ones((3, 3), dtype=np.int32)
        # Coverage matrix:
        coverage_matrix = np.zeros((len(distance), num_x, num_y), dtype=np.int32)
        for dev_idx, device_bool in enumerate((coverage_boolean == 1).T):
            _coord = coordinates_index[device_bool]
            coverage_matrix[dev_idx, _coord[:, 0], _coord[:, 1]] = 1
        self.coverage = coverage_matrix
        # Dimension and bounds:
        self.dimensions = np.sum(device_matrix) * len(distance)
        self.bounds = (0, 1)
        self.distances = distance // min(delta_x, delta_y)
        # Set up fitness function:
        if framework == 'numpy' or framework == 'np' or framework == 'npy':
            self.__call = self.fitness_function_np
        elif framework == 'tensorflow' or framework == 'tf':
            self.__call = tf.function(self.fitness_function_np)
        else:
            raise ValueError(f"Framework not supported: {framework}")

        # Pre-computed values:
        self._expanded_base = np.repeat(self.base_d[np.newaxis, :, :], self.coverage.shape[0], axis=0)
        self.__penalty__ = 1_000_000
        self.__threshold_density__ = 0.9
        self.__coverage_density__ = 0.3

        # Coverage computation:
        _expanded_cov = np.mean(self.coverage, axis=0)
        _expanded_cov = np.where(_expanded_cov >= self.__coverage_density__, 1, 0)
        self.coverage = _expanded_cov

    # Numpy function:
    def fitness_function_np(self, individuals: np.ndarray) -> np.ndarray:
        """
        This function calculates the fitness of each individual.
        :param individuals: Array of individuals.
        :return: Array of fitness values.
        """
        # Global fitness parameters:
        __penalty__ = self.__penalty__
        __threshold_density__ = self.__threshold_density__
        __coverage_density__ = self.__coverage_density__
        # Set main variables:
        population = np.where(individuals >= __threshold_density__, 1, 0)
        _d = np.zeros((population.shape[0], self.cost.shape[0], self.z.shape[0], self.z.shape[1]))
        _expanded_base = self._expanded_base
        _expanded_base = np.repeat(_expanded_base[np.newaxis, :, :, :], population.shape[0], axis=0)
        _expanded_z = np.repeat(self.z[np.newaxis, :, :], population.shape[0], axis=0)
        # Expanded coverage for all individuals:
        _expanded_cov = np.repeat(self.coverage[np.newaxis, :, :], population.shape[0], axis=0)
        # Population [NI, NP]
        # Pos [NP, 3]
        # Mat [NI, ND, NX, NY]
        np.place(_d, _expanded_base, population)
        # Initial cost based on cost matrix:
        _cost_ = (np.repeat(self.cost[:, np.newaxis], int(population.shape[-1] / self.cost.shape[0]), axis=-1).
                  reshape((1, population.shape[-1])))
        cost = np.sum(population * _cost_, axis=-1)
        # Convolve:
        # Di+1 = (Di * K) and Z
        for _dev_idx, _cost_ref in enumerate(self.cost):
            _dev = _d[:, _dev_idx, :, :]
            for _ in range(int(self.distances[_dev_idx])):
                _dev = conv2d(_dev, self.k)
                _dev = np.logical_and(_dev, _expanded_z).astype(np.int32)
            _d[:, _dev_idx, :, :] = _dev
        # Compute cost:
        _collapsed_along_dev_ = np.sum(_d, axis=1)
        _clipped_cost = np.clip(_expanded_cov - _collapsed_along_dev_, 0, 1)
        _clipped_cost_along_axis = np.sum(_clipped_cost, axis=(1, 2))
        cost += (__penalty__ * _clipped_cost_along_axis).astype(np.int32)
        # As type np.int32
        return -cost.astype(np.int32)

    def plot_solution(self, individual: (tf.Tensor, np.array), save: bool = False,
                      path: str = './optimization.png') -> None:
        """
        This function plots the solution.
        :param individual: The individual to plot.
        :param save: If the image is going to be saved.
        :param path: Path to save the image.
        :return: Nothing.
        """
        # Global fitness parameters:
        __penalty__ = self.__penalty__
        __threshold_density__ = self.__threshold_density__
        __coverage_density__ = self.__coverage_density__
        # Set main variables:
        population = np.where(individual >= __threshold_density__, 1, 0)
        _d = np.zeros((self.cost.shape[0], self.z.shape[0], self.z.shape[1]))
        _expanded_base = self._expanded_base
        _expanded_z = self.z
        # Expanded coverage for all individuals:
        _expanded_cov = self.coverage
        # Population [NI, NP]
        # Pos [NP, 3]
        # Mat [NI, ND, NX, NY]
        np.place(_d, _expanded_base, population)
        # Initial cost based on cost matrix:
        _cost_ = (np.repeat(self.cost[:, np.newaxis], int(population.shape[-1] / self.cost.shape[0]), axis=-1).
                  reshape((1, population.shape[-1])))
        cost = np.sum(population * _cost_, axis=-1)
        # Convolve:
        # Di+1 = (Di * K) and Z
        old_devs = _d.copy()
        for _dev_idx, _cost_ref in enumerate(self.cost):
            _dev = _d[_dev_idx, :, :]
            for _ in range(int(self.distances[_dev_idx])):
                _dev = conv2d(np.expand_dims(_dev, axis=0), self.k)[0]
                _dev = np.logical_and(_dev, _expanded_z).astype(np.int32)
            _d[_dev_idx, :, :] = _dev
        # Compute cost:
        _collapsed_along_dev_ = np.sum(_d, axis=0)
        _clipped_cost = np.clip(_expanded_cov - _collapsed_along_dev_, 0, 1)
        _clipped_cost_along_axis = np.sum(_clipped_cost)
        cost += (__penalty__ * _clipped_cost_along_axis).astype(np.int32)
        # Plot:
        plt.figure(figsize=(10, 5))
        # For each device one subplot:
        for dev in range(_d.shape[0]):
            plt.subplot(1, len(self.cost), dev + 1)
            # _d[dev] to RGB:
            x = _d[dev, :, :, np.newaxis] * np.array([0, 255, 0])
            y = np.expand_dims(_expanded_cov, axis=-1) * np.array([255, 0, 0])
            z = old_devs[dev, :, :, np.newaxis] * np.array([-255, -255, 0])
            plt.imshow(x + y + z)
            plt.title(f'Device {dev + 1}')
        plt.suptitle(f'Cost: {cost[0]}')
        if save:
            plt.savefig(path)

    def __call__(self, *args, **kwargs):
        return self.__call(*args, **kwargs)


def main() -> None:
    logging.info("[+] Connected to Coverage Problem.")
    # Load csv file:
    coordinates, is_position, coverage_boolean, distance, cost = format_array(*csv_to_numpy(MAP_PATH, DEVICE_PATH))
    fitness_function = Fitness(coordinates, is_position, coverage_boolean, distance, cost, framework='numpy')
    logging.info(f"[!] Fitness built successfully. Setting up TensorCro...")
    # [!] Set up TensorCro:
    # - Main parameters:
    n_dims = len(fitness_function.device_coordinates)
    directives = tf.convert_to_tensor([[fitness_function.bounds[0]] * n_dims,
                                       [fitness_function.bounds[1]] * n_dims],
                                      dtype_hint=tf.float32)
    reef_shape = (50, 40)
    # - Substrates:
    uniform_crossover = UniformCrossover()
    harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=directives)
    random_search = RandomSearch(directives, 0.2)
    genetic_algorithm = ComposedSubstrate(MultipointCrossover([n_dims // 2]),
                                          Mutation('gaussian', mean=0.0, stddev=0.05),
                                          name='GeneticAlgorithm')
    subs = [uniform_crossover, harmony_search, random_search, genetic_algorithm]
    # - TensorCro:
    t_cro = TensorCro(reef_shape, subs=subs)
    # - Fit:
    logging.info(f"[!] TensorCro built successfully. Starting optimization...")
    # best = t_cro.fit(fitness_function, directives, max_iter=200, device='/GPU:0', seed=0, shards=5, save=True)
    fitness_function(np.random.uniform(0, 1, (100, n_dims)))
    best = np.load('./sol.npy')
    logging.info(f'[!] Optimization finished. Best individual: {best}')
    # - Plot:
    fitness_function.plot_solution(best[0], save=True, path='./optimization.png')
    logging.info("[-] Disconnected from Coverage Problem.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
