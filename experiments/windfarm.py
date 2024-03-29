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
import pandas as pd
import matplotlib.pyplot as plt
from other_algorithms import GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm
from tensorcro import TensorCro, MultipointCrossover, ParticleSwarmOptimization, HarmonySearch, SimulatedAnnealing, \
    ComposedSubstrate, Mutation, LightCallback, CoordinateDescent
from windfarm_problem import (fitness_tf, TurbineParameters, WindParameters)
logging.basicConfig(level=logging.INFO)
# SLACK_TOKEN = 'xoxb-5951328403522-6081308292118-YvKWkLfSrLGMtEBskZ6ycbWE'
ALGORITHMS = [GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm]
TIME_LIMIT = 60 * 60
SCBK = False
SEEDS = [2023 + i for i in range(10)]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Fitness:
    def __init__(self, dmin: float = TurbineParameters.min_dist, reverse=False):
        """
        Fitness function for the wind farm problem.
        :param dmin: Minimum distance between turbines.
        """
        self.dmin = tf.constant(dmin, dtype=tf.float32)
        self.fitness_tf = fitness_tf
        self.reverse = 1 if not reverse else -1

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
        # Store input type:
        if isinstance(args[0], np.ndarray):
            input_type = np.ndarray
        else:
            input_type = tf.Tensor
        # Transform polar coordinates to cartesian:
        cartesian_pop = tf.cast(self.polar_to_cartesian(args[0]), tf.float32)
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
        if input_type == np.ndarray:
            retval = (total_power - penalty).numpy() * self.reverse
        else:
            retval = (total_power - penalty) * self.reverse
        return retval


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

        # - Substrates:
        nsubs = 5
        nrows = 200 // (2 * nsubs)
        reef_shape = (nrows, nsubs * 2)
        pso = ParticleSwarmOptimization(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
        harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.2, bandwidth=0.27, directives=directives)
        coordinate_descent = CoordinateDescent(directives, number_of_divs=20)
        genetic_algorithm = ComposedSubstrate(MultipointCrossover([directives.shape[-1] // 2]),
                                              Mutation('gaussian', mean=0.0, stddev=0.27),
                                              name='GeneticAlgorithm')
        simulated_annealing = SimulatedAnnealing(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
        subs = [pso, harmony_search, genetic_algorithm, simulated_annealing, coordinate_descent]

        # - TensorCro:
        t_cro = TensorCro(reef_shape, subs=subs)

        # - Fit:
        logging.info(f"\n[!] TensorCro built successfully. Starting optimization...")
        for seed in SEEDS:
            if not os.path.exists(f'./results/windfarm/pop/best_solutions_{seed}_{dmax}m.npy'):
                logging.info(f"[!] Starting TensorCro: {seed}:{dmax}m...")
                if SCBK:
                    slack_callback = LightCallback(verbose=True)
                else:
                    slack_callback = None
                try:
                    best = t_cro.fit(fitness_function, directives, max_iter=50_000, device='/GPU:0', seed=seed,
                                     shards=10_000, save=False, time_limit=TIME_LIMIT, tf_compile=True,
                                     callback=slack_callback, minimize=False)
                    fitness_function.plot_solution(best[0], int(dmax), save=True,
                                                   path=f'./results/windfarm/render/optimization_tensorcro_'
                                                        f'{seed}_{dmax}.png')
                except Exception as ex:
                    logging.error(f"[!] TensorCro failed.")
                    if slack_callback is not None:
                        slack_callback.exception_handler(ex)
                    raise ex
                if slack_callback is not None:
                    slack_callback.end(best[0].numpy())
                np.save(f'./results/windfarm/pop/best_solutions_{seed}_{dmax}m.npy', best[0].numpy())
                np.save(f'./results/windfarm/fitness/best_fitness_{seed}_{dmax}m.npy', best[1].numpy())
                logging.info(f'[!] Optimization finished. Best individual: {best}')

        if not os.path.exists(f'./results/windfarm/pop/best_solutions_long_{dmax}m.npy'):
            logging.info(f"[!] Starting TensorCro: Long run:{dmax}m...")
            if SCBK:
                slack_callback = LightCallback(verbose=True)
            else:
                slack_callback = None
            try:
                best = t_cro.fit(fitness_function, directives, max_iter=10_000_000, device='/GPU:0', seed=2000,
                                 shards=1_000_000, save=False, time_limit=TIME_LIMIT, tf_compile=True,
                                 callback=slack_callback, minimize=False)
                fitness_function.plot_solution(best[0], int(dmax), save=True,
                                               path=f'./results/windfarm/render/optimization_tensorcro_'
                                                    f'long_{dmax}.png')
            except Exception as ex:
                logging.error(f"[!] TensorCro failed.")
                if slack_callback is not None:
                    slack_callback.exception_handler(ex)
                raise ex

            if slack_callback is not None:
                slack_callback.end(best[0].numpy())
            np.save(f'./results/windfarm/pop/best_solutions_long_{dmax}m.npy', best[0].numpy())
            np.save(f'./results/windfarm/fitness/best_fitness_long_{dmax}m.npy', best[1].numpy())
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


def timed_main():
    """
    Run all algorithms and experiments 1h.
    :return:
    """
    logging.info("[+] Connected to Wind Farm Problem.")
    dmin = TurbineParameters.min_dist

    for seed in SEEDS:
        for dmax, nturb in zip(WindParameters.park_diam, [16, 36, 64]):
            # [!] Set up TensorCro:
            # - Main parameters:
            fitness_function = Fitness(dmin)
            logging.info(f"[!] Fitness built successfully. Setting up TensorCro...")
            first_half = [dmax] * nturb
            second_half = [2 * np.pi] * nturb
            concatenated_max = np.concatenate([first_half, second_half])
            concatenated_min = np.array([0] * (2 * nturb))
            directives = tf.convert_to_tensor([concatenated_min, concatenated_max], dtype_hint=tf.float32)

            # - Substrates:
            nsubs = 5
            nrows = 200 // (2 * nsubs)
            reef_shape = (nrows, nsubs * 2)
            pso = ParticleSwarmOptimization(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
            harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.2, bandwidth=0.27, directives=directives)
            coordinate_descent = CoordinateDescent(directives, number_of_divs=20)
            genetic_algorithm = ComposedSubstrate(MultipointCrossover([directives.shape[-1] // 2]),
                                                  Mutation('gaussian', mean=0.0, stddev=0.27),
                                                  name='GeneticAlgorithm')
            simulated_annealing = SimulatedAnnealing(directives, shape=(reef_shape[0], reef_shape[1] // nsubs))
            subs = [pso, harmony_search, genetic_algorithm, simulated_annealing, coordinate_descent]

            # - TensorCro:
            t_cro = TensorCro(reef_shape, subs=subs)

            # - Fit:
            logging.info(f"\n[!] TensorCro built successfully. Starting optimization...")

            if not os.path.exists(f'./results/windfarm1h/pop/best_1h_TensorCRO_{seed}_{dmax}m.npy'):
                logging.info(f"[!] Starting TensorCro: {seed}:{dmax}m...")
                if SCBK:
                    slack_callback = LightCallback(verbose=True)
                else:
                    slack_callback = None
                try:
                    best = t_cro.fit(fitness_function, directives, max_iter=500_000, device='/GPU:0', seed=seed,
                                     shards=100_000, save=False, time_limit=TIME_LIMIT, tf_compile=True,
                                     callback=slack_callback, minimize=False)
                    fitness_function.plot_solution(best[0], int(dmax), save=True,
                                                   path=f'./results/windfarm1h/render/optimization_tensorcro_1h_'
                                                        f'{seed}_{dmax}.png')
                except Exception as ex:
                    logging.error(f"[!] TensorCro failed.")
                    if slack_callback is not None:
                        slack_callback.exception_handler(ex)
                    raise ex
                if slack_callback is not None:
                    slack_callback.end(best[0].numpy())
                np.save(f'./results/windfarm1h/pop/best_1h_TensorCRO_{seed}_{dmax}m.npy', best[0].numpy())
                np.save(f'./results/windfarm1h/fitness/best_fitness_1h_TensorCRO_{seed}_{dmax}m.npy', best[1].numpy())
                logging.info(f'[!] Optimization finished. Best individual: {best}')

            # - Other algorithms:
            for algorithm in ALGORITHMS:
                if not os.path.exists(f'./results/windfarm1h/pop/best_1h_{algorithm.__name__}_{seed}_{dmax}m.npy'):
                    logging.info(f"[!] Starting {algorithm.__name__}: {seed}:{dmax}m...")
                    try:
                        fitness_function = Fitness(dmin, reverse=True)
                        if algorithm is not SimulatedAnnealingAlgorithm:
                            ai = algorithm(200)
                        else:
                            ai = algorithm()
                        # We run the algorithm:
                        best_ind, best_fit = ai.fit(fitness_function, directives.numpy(), int(1e10),
                                                    time_limit=TIME_LIMIT, seed=seed, verbose=True)
                        if isinstance(best_ind, tf.Tensor):
                            best_ind = best_ind.numpy()
                            best_fit = best_fit.numpy()
                        else:
                            best_ind = np.array(best_ind)
                            best_fit = np.array(best_fit)
                        np.save(f'./results/windfarm1h/pop/best_1h_{algorithm.__name__}_{seed}_{dmax}m.npy',
                                best_ind)
                        np.save(f'./results/windfarm1h/fitness/best_fitness_1h_{algorithm.__name__}_{seed}_{dmax}m.npy',
                                best_fit)
                        fitness_function.plot_solution(best_ind[0], dmax, save=True,
                                                       path=f'./results/windfarm1h/render/'
                                                            f'optimization_1h_{algorithm.__name__}_{seed}.png')
                        logging.info(f'[!] Optimization finished. Best fitness: {best_fit}')
                    except Exception as ex:
                        logging.error(f"[!] {algorithm.__name__} failed: {ex}")
        logging.info("[-] Disconnected from Wind Farm Problem.")


def read():
    _lsta = os.listdir('./experiments/')
    lsta = list()
    for _ in _lsta:
        if 'fitness_1h' in _:
            lsta.append(_)
    del _lsta
    _lsta2 = os.listdir('./experiments/results/windfarm/fitness')
    lsta2 = list()
    for _ in _lsta2:
        if 'fitness_1h' in _:
            lsta2.append(_)
    del _lsta2
    results = {'alg': list(), 'seed': list(), 'best': list(), 'scenario': list()}
    for end_path in lsta:
        path = './experiments/' + end_path
        best = np.max(np.load(path))
        name = end_path.split('_')[3]
        seed = end_path.split('_')[4]
        scen = end_path.split('_')[5][:5]
        results['alg'].append(name)
        results['seed'].append(seed)
        results['best'].append(best)
        results['scenario'].append(scen)
    for end_path in lsta2:
        path = './experiments/results/windfarm/fitness/' + end_path
        best = np.max(np.load(path))
        name = 'TensorCRO'
        seed = end_path.split('_')[3]
        scen = end_path.split('_')[4][:5]
        results['alg'].append(name)
        results['seed'].append(seed)
        results['best'].append(best)
        results['scenario'].append(scen)
    rdf = pd.DataFrame(results)
    maxs = rdf.groupby(['alg', 'scenario']).max()


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    timed_main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
