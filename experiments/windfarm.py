# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from other_algorithms import GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm
from tensorcro import TensorCro, UniformCrossover, ParticleSwarmOptimization, HarmonySearch, SimulatedAnnealing, \
    RandomSearch, ComposedSubstrate, Mutation, SlackCallback
from tensorcro.substrates import CROSubstrate
from windfarm_problem import (fitness_tf, TurbineParameters, WindParameters)
logging.basicConfig(level=logging.INFO)
SLACK_TOKEN = 'xoxb-5951328403522-6081308292118-YvKWkLfSrLGMtEBskZ6ycbWE'
ALGORITHMS = [GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm]
TIME_LIMIT = None
SCBK = False
SEEDS = [2023 + i for i in range(10)]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class RandomPermutation(CROSubstrate):
    def __init__(self, section_points: int):
        """
        This substrate performs a random permutation of the parameters.
        :param section_points: Section points of the parameters.
        """
        self.section_points = tf.constant(section_points, dtype=tf.int32)

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        # Empty vector of shape [0, 2 * nturbines]:
        permuted_shaped = None
        # Permute individuals so we have a 2D tensor with each individual and its permuted parameters:
        multip = tf.shape(individuals)[-1] // self.section_points
        for individual in individuals:
            # We create a permutation tensor:
            permutation_tensor = tf.random.shuffle(tf.range(multip))
            permutation_tensor = tf.cast(permutation_tensor, tf.float32)
            permutation_tensor = tf.expand_dims(permutation_tensor, axis=0)
            permutation_tensor = tf.tile(permutation_tensor, [self.section_points, 1])
            # Reshape to [1, -1]:
            permutation_tensor = tf.cast(tf.reshape(permutation_tensor, [1, -1]), tf.int32)
            # Add [0, 0, 0, ...] until tf.shape[-1] // 2 and [1, 1, 1, 1, ...] until from tf.shape[-1] //
            # 2 to tf.shape[-1]:
            permutations_add = tf.concat([tf.zeros(shape=(1, multip), dtype=tf.int32), multip * tf.ones(
                shape=(1, multip), dtype=tf.int32)], axis=-1)
            # We add the values:
            permutations = permutation_tensor + permutations_add
            # We permute the individual:
            individual_out_unshaped = tf.gather(individual, permutations, axis=-1)
            # We append the individual to the list:
            if permuted_shaped is not None:
                permuted_shaped = tf.concat([permuted_shaped, individual_out_unshaped], axis=0)
            else:
                permuted_shaped = individual_out_unshaped
        # Remove all the permutated elements that are the same that in the original individual:
        repeated_index = tf.reduce_all(tf.equal(permuted_shaped, individuals), axis=-1)
        # Choose only false values:
        repeated_index = tf.logical_not(repeated_index)
        # Get indices where true:
        repeated_index = tf.where(repeated_index)
        # We filter the individuals:
        retval = tf.gather(individuals, repeated_index, axis=0)
        # We return the individuals:
        return tf.squeeze(retval)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Fitness:
    def __init__(self, dmin: float = TurbineParameters.min_dist):
        """
        Fitness function for the wind farm problem.
        :param dmin: Minimum distance between turbines.
        """
        self.dmin = tf.constant(dmin, dtype=tf.float32)
        self.fitness_tf = fitness_tf

    def plot_solution(self, individual: (tf.Tensor, np.array), diam: int, save: bool = False,
                      path: str = './optimization.png') -> None:
        """
        Plots the solution of the problem.
        :param individual: Polar coordinates of the turbines.
        :param diam: Diameter of the wind farm.
        :param save: Whether to save the plot or not.
        :param path: Path to save the plot.
        :return: Nothing.
        """
        # We transform the polar coordinates to cartesian:
        if isinstance(individual, np.ndarray):
            cartesian_coords = self.polar_to_cartesian(tf.convert_to_tensor(individual, dtype_hint=tf.float32)).numpy()
        else:
            cartesian_coords = self.polar_to_cartesian(individual).numpy()
        cartesian_coords = cartesian_coords[0].T
        # We plot the solution:
        plt.figure(figsize=(10, 10))
        plt.scatter(cartesian_coords[:, 0], cartesian_coords[:, 1], s=100, c='r', marker='x')
        plt.xlim(-1.1 * diam, 1.1 * diam)
        plt.ylim(-1.1 * diam, 1.1 * diam)
        plt.title(f'Wind Farm Problem (Scenario {diam}m)')
        # Plot a big circle:
        circle = plt.Circle((0, 0), diam, color='k', fill=False, linestyle='--')
        plt.gcf().gca().add_artist(circle)
        # Plot a circle with center each individual:
        for i in range(cartesian_coords.shape[0]):
            circle = plt.Circle((float(cartesian_coords[i, 0]), float(cartesian_coords[i, 1])), self.dmin / 2,
                                color='b', fill=False, linestyle='--')
            plt.gcf().gca().add_artist(circle)
        if save:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def polar_to_cartesian(individual: (tf.Tensor, np.array)) -> tf.Tensor:
        """
        Transforms the polar coordinates of the turbines to cartesian coordinates.
        :param individual: Individual to transform. (first half distances, second half angles)
        :return:
        """
        # We split the individual in two:
        nturb = tf.shape(individual)[-1] // 2
        distances = individual[:, :nturb]
        angles = individual[:, nturb:]

        # We compute the cartesian coordinates:
        x = tf.math.cos(angles) * distances
        y = tf.math.sin(angles) * distances

        # We return the cartesian coordinates:
        return tf.stack([x, y], axis=1)

    def __call__(self, *args, **kwargs):
        # Transform polar coordinates to cartesian:
        cartesian_pop = self.polar_to_cartesian(args[0])
        # [1] Compute penalty if there are turbines too close:
        # Expand dimensions to make broadcasting work
        cartesian_array_expanded_1 = tf.expand_dims(cartesian_pop, axis=-1)
        cartesian_array_expanded_2 = tf.expand_dims(cartesian_pop, axis=-2)
        # Compute pairwise differences
        pairwise_differences = cartesian_array_expanded_1 - cartesian_array_expanded_2
        # Compute pairwise distances using tf.norm
        distances = tf.norm(pairwise_differences, axis=1)
        # Add affinity to the diagonal
        distances = tf.linalg.set_diag(distances, tf.ones(shape=(tf.shape(distances)[:-1]), dtype=tf.float32) *
                                       2 * self.dmin)
        # Add penalty if there are turbines too close
        penalty = tf.reduce_sum(tf.where(distances < self.dmin, 1e20, 0.0), axis=-1)
        penalty = tf.reduce_sum(penalty, axis=-1)
        # [2] We calculate the fitness value:
        windfram_power = self.fitness_tf(tf.convert_to_tensor(cartesian_pop, dtype_hint=tf.float32))
        total_power = tf.reduce_sum(windfram_power, axis=-1)
        return total_power - penalty


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main() -> None:
    logging.info("[+] Connected to Wind Farm Problem.")

    dmin = TurbineParameters.min_dist
    fitness_function = Fitness(dmin)
    logging.info(f"[!] Fitness built successfully. Setting up TensorCro...")

    for dmax, nturb in zip(WindParameters.park_diam, [16, 36, 64]):
        # [!] Set up TensorCro:
        # - Main parameters:
        first_half = [dmax] * nturb
        second_half = [2 * np.pi] * nturb
        concatenated_max = np.concatenate([first_half, second_half])
        concatenated_min = np.array([0] * (2 * nturb))
        directives = tf.convert_to_tensor([concatenated_min, concatenated_max], dtype_hint=tf.float32)
        reef_shape = (20, 50)

        # - Substrates:
        nsubs = 5
        pso = ParticleSwarmOptimization(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
        harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=directives)
        random_search = RandomSearch(directives, 0.2)
        genetic_algorithm = ComposedSubstrate(UniformCrossover(),
                                              Mutation('gaussian', mean=0.0, stddev=0.05),
                                              name='GeneticAlgorithm')
        simulated_annealing = SimulatedAnnealing(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
        rp = RandomPermutation(2)
        subs = [pso, harmony_search, random_search, genetic_algorithm, simulated_annealing]

        # - TensorCro:
        t_cro = TensorCro(reef_shape, subs=subs)

        # - Fit:
        logging.info(f"[!] TensorCro built successfully. Starting optimization...")
        for seed in SEEDS:
            if not os.path.exists(f'./results/windfarm/best_solutions_{seed}_{dmax}m.npy'):
                logging.info(f"[!] Starting TensorCro: {seed}:{dmax}m...")
                if SCBK:
                    slack_callback = SlackCallback(SLACK_TOKEN, '#tensor-cro-dev', 'Alverciito',
                                                   simulation_name=f'Wind Farm Problem (Scenario {dmax}m)')
                else:
                    slack_callback = None
                try:
                    best = t_cro.fit(fitness_function, directives, max_iter=50_000, device='/GPU:0', seed=seed,
                                     shards=1_000, save=False, time_limit=TIME_LIMIT, tf_compile=False,
                                     callback=slack_callback,
                                     minimize=False)
                    fitness_function.plot_solution(best[0], int(dmax), save=True,
                                                   path=f'./optimization_tensorcro_{seed}.png')
                except Exception as ex:
                    logging.error(f"[!] TensorCro failed.")
                    if slack_callback is not None:
                        slack_callback.exception_handler(ex)
                    raise ex
                if slack_callback is not None:
                    slack_callback.end(best[0].numpy())
                np.save(f'./results/windfarm/best_solutions_{seed}_{dmax}m.npy', best[0].numpy())
                np.save(f'./results/windfarm/best_fitness_{seed}_{dmax}m.npy', best[1].numpy())
                logging.info(f'[!] Optimization finished. Best individual: {best}')

            # - Other algorithms:
            # for algorithm in ALGORITHMS:
            #     if not os.path.exists(f'./best_solutions_{algorithm.__name__}_{seed}_{dmax}m.npy'):
            #         logging.info(f"[!] Starting {algorithm.__name__}: {seed}:{dmax}m...")
            #         try:
            #             if algorithm is not SimulatedAnnealingAlgorithm:
            #                 ai = algorithm(200)
            #             else:
            #                 ai = algorithm()
            #             # We run the algorithm:
            #             best_ind, best_fit = ai.fit(fitness_function, directives.numpy(), int(1e10),
            #                                         time_limit=TIME_LIMIT, seed=seed)
            #         except Exception as ex:
            #             logging.error(f"[!] {algorithm.__name__} failed.")
            #             raise ex
            #         np.save(f'./best_solutions_{algorithm.__name__}_{seed}_{dmax}m.npy', best_ind)
            #         np.save(f'./best_fitness_{algorithm.__name__}_{seed}_{dmax}m.npy', best_fit)
            #         fitness_function.plot_solution(best_ind[0], dmax, save=True,
            #                                        path=f'./optimization_{algorithm.__name__}_{seed}.png')
            #         logging.info(f'[!] Optimization finished. Best fitness: {best_fit}')
        logging.info("[-] Disconnected from Wind Farm Problem.")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
