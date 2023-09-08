# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import time
import numpy as np
import logging
# Algorithms:
from tensorcro import TensorCro, UniformCrossover, MultipointCrossover, HarmonySearch, RandomSearch, \
    ComposedSubstrate, Mutation, DifferentialSearch
from other_algorithms import GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm
# Many local minima functions:
from test_functions import AckleyFunction, BukinFunction6, DropWaveFunction, EggHolderFunction, \
        GramacyLeeFunction, GrieWankFunction, HolderTableFunction, RastriginFunction
# Bowl shape functions:
from test_functions import HyperEpsiloid, HyperSphere
# Valley shape functions:
from test_functions import DixonPrice, Rosenbrock
# Flat shape functions:
from test_functions import Easom, Michalewicz
# Other curious functions:
from test_functions import StyblinskiTang, Powell
# Set logger to results.log and console:
logging.basicConfig(filename='./results/logging.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(format='[%(asctime)s] ^ %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# Global parameters:
SEEDS = [2023 + i for i in range(10)]
FUNCTIONS = {
    'local_minima': [
        AckleyFunction, BukinFunction6, DropWaveFunction, EggHolderFunction,
        GramacyLeeFunction, GrieWankFunction, HolderTableFunction, RastriginFunction],
    'bowl_shape':
        [HyperEpsiloid, HyperSphere],
    'valley_shape':
        [DixonPrice, Rosenbrock],
    'flat_shape':
        [Easom, Michalewicz],
    'other':
        [StyblinskiTang, Powell]}
ALGORITHMS = [TensorCro, GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm]
DIMENSIONS = [5, 20, 50, 200, 500]
DIMENSIONS.reverse()
RESULTS_PATH = './results/'
TIME_LIMIT = 100


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def run_tests() -> None:
    # We run the test 1 for the article:
    logging.info('[+] Connected to Test 1:')
    test_1()
    logging.info('[-] Disconnected to Test 1.\n\n')
    # # We run the test 2 for the article:
    # logging.info('[+] Connected to Test 2:')
    # test_2()
    # logging.info('[-] Disconnected to Test 2.\n\n')
    # # We run the test 3 for the article:
    # logging.info('[+] Connected to Test 3:')
    # test_3()
    # logging.info('[-] Disconnected to Test 3.\n\n')
    # # We run the test 4 for the article:
    # logging.info('[+] Connected to Test 4:')
    # test_4()
    # logging.info('[-] Disconnected to Test 4.\n\n')


def test_1() -> None:
    # We run the test 1 with the following parameters:
    # - Seed number:            2023, 2024, 2025, 2026, 2027, ..., 2033. (10 seeds)
    # - Number of iterations:   Time limited (2 - 5 - 10) minutes.
    # - Number of individuals:  Dimension dependent: 20 times the dimension.
    # - Number of dimensions:   Variable.
    # - Function:               All implemented functions.
    # - Algorithms:             All implemented algorithms.

    # Read current results:
    try:
        with open(RESULTS_PATH + 'test_1.json', 'r') as f:
            current_json = json.load(f)
    except FileNotFoundError:
        current_json = {'test_1': list()}
    except json.decoder.JSONDecodeError:
        current_json = {'test_1': list()}

    # We run the test:
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            for function_type, function_list in FUNCTIONS.items():
                for test_function in function_list:
                    for algorithm in ALGORITHMS:
                        if not check_occurrence({'seed': seed, 'dimension': dimension, 'function':
                                                 test_function.__name__, 'algorithm': algorithm.__name__},
                                                current_json['test_1']):
                            # We compose the current test:
                            logging.info(f'[+] Running test with seed {seed}, dimension {dimension}, '
                                         f'function {test_function.__name__} and algorithm {algorithm.__name__}.')
                            test_function_instance = test_function()
                            core_bounds = test_function_instance.bounds
                            bounds = np.array([core_bounds] * dimension, dtype=np.float32).T
                            # Run the algorithm:
                            tik = time.perf_counter()
                            if algorithm is not SimulatedAnnealingAlgorithm and algorithm is not TensorCro:
                                ai = algorithm(20 * dimension)
                                # We run the algorithm:
                                best_ind, best_fit = ai.fit(test_function_instance, bounds, int(1e10),
                                                            time_limit=TIME_LIMIT, seed=seed)
                            elif algorithm is TensorCro:
                                uniform_crossover = UniformCrossover()
                                harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=bounds)
                                random_search = RandomSearch(bounds, 0.2)
                                genetic_algorithm = ComposedSubstrate(
                                    MultipointCrossover([dimension // 2]),
                                    Mutation('gaussian', mean=0.0, stddev=0.05), name='GeneticAlgorithm'
                                )
                                differential_search = DifferentialSearch(bounds)
                                subs = [uniform_crossover, harmony_search, random_search,
                                        genetic_algorithm, differential_search]
                                ai = TensorCro(reef_shape=(5, dimension * 4), subs=subs)
                                # We run the algorithm:
                                best_ind, best_fit = ai.fit(test_function_instance, bounds,
                                                            max_iter=int(1e6), save=False,
                                                            time_limit=TIME_LIMIT, seed=seed,
                                                            shards=1, minimize=True)
                            else:
                                ai = algorithm()
                                # We run the algorithm:
                                best_ind, best_fit = ai.fit(test_function_instance, bounds, int(1e10),
                                                            time_limit=TIME_LIMIT, seed=seed)
                            tok = time.perf_counter()
                            # We convert the results to list if they are not:
                            if not isinstance(best_ind, np.ndarray):
                                best_ind = best_ind.numpy().tolist()
                                best_fit = best_fit.numpy().tolist()
                            # Save the results:
                            with open(RESULTS_PATH + 'test_1.json', 'w') as f:
                                current_json['test_1'].append({
                                    'algorithm': algorithm.__name__,
                                    'function': test_function.__name__,
                                    'function_type': function_type,
                                    'dimension': dimension,
                                    'best_fit': float(best_fit[0]),
                                    'best_ind': list(best_ind[0]),
                                    'elapsed_time': tok - tik,
                                    'seed': seed,
                                    'num_eval': test_function_instance.number_of_evaluations})
                                json.dump(current_json, f, indent=4)
                                f.write('\n')
                            # Logging:
                            logging.info(f'Test 1:\nAlgorithm: {algorithm.__name__},\n'
                                         f'Function: {test_function.__name__}:'
                                         f'{test_function_instance.number_of_evaluations} eval,\n'
                                         f'Elapsed time: {tok - tik},'
                                         f'\nDimension: {dimension},\nBest fitness: {best_fit[0]},\nBest individual:'
                                         f'{best_ind[0]}')


# def test_2() -> None:
#     # We run the test 2 with the following parameters:
#     # - Seed number:            2023
#     # - Number of iterations:   10_000 fitness evaluations.
#     # - Number of individuals:  Dimension dependent: 20 times the dimension.
#     # - Number of dimensions:   Variable.
#     # - Function:               All implemented functions.
#     # - Algorithms:             All implemented algorithms.
#     for algorithm in ALGORITHMS:
#         for function_type, function_list in FUNCTIONS.items():
#             for test_function in function_list:
#                 for dimension in DIMENSIONS:
#                     # Run the algorithm:
#                     if algorithm is not SimulatedAnnealingAlgorithm:
#                         ai = algorithm(20 * dimension)
#                     else:
#                         ai = algorithm()
#                     # We run the algorithm:
#                     test_function_instance = test_function()
#                     core_bounds = test_function_instance.bounds
#                     bounds = np.array([core_bounds] * dimension).T
#                     best_ind, best_fit = ai.fit(test_function_instance, bounds, -1, seed=SEED)
#                     # Save the results:
#                     with open(RESULTS_PATH + 'test_2.json', 'a') as f:
#                         json.dump({'algorithm': algorithm.__name__,
#                                    'function': test_function.__name__,
#                                    'dimension': dimension,
#                                    'best_fit': float(best_fit),
#                                    'best_ind': list(best_ind),
#                                    'num_eval': test_function_instance.number_of_evaluations}, f, indent=4)
#                         f.write('\n')
#                     # Logging:
#                     logging.info(f'Test 2: Algorithm: {algorithm.__name__}, Function: {test_function.__name__}:'
#                                  f'{test_function_instance.number_of_evaluations} eval, '
#                                  f'Dimension: {dimension}, Best fitness: {best_fit[0]}, Best individual:
#                                  {best_ind[0]}')
#
#
# def test_3() -> None:
#     # We run the test 2 with the following parameters:
#     # - Seed number:            2023
#     # - Number of iterations:   10 minutes and 50k fitness evaluations per function.
#     # - Number of individuals:  Dimension dependent: 20 times the dimension.
#     # - Number of dimensions:   Variable.
#     # - Function:               All implemented functions.
#     # - Algorithms:             TensorCRO(GPU), CRO(CPU).
#     for device in ['/GPU:0', '/CPU:0']:
#         for function_type, function_list in FUNCTIONS.items():
#             for test_function in function_list:
#                 for dimension in DIMENSIONS:
#                     # Run the algorithm:
#                     ai = TensorCro()
#                     # We run the algorithm:
#                     test_function_instance = test_function()
#                     core_bounds = test_function_instance.bounds
#                     bounds = np.array([core_bounds] * dimension).T
#                     best_ind, best_fit = ai.fit(test_function_instance, bounds, -1, seed=SEED, device=device)
#                     # Save the results:
#                     with open(RESULTS_PATH + 'test_3.json', 'a') as f:
#                         json.dump({'algorithm': f'TensorCRO:{device}',
#                                    'function': test_function.__name__,
#                                    'dimension': dimension,
#                                    'best_fit': float(best_fit),
#                                    'best_ind': list(best_ind),
#                                    'num_eval': test_function_instance.number_of_evaluations}, f, indent=4)
#                         f.write('\n')
#                     # Logging:
#                     logging.info(f'Test 3: Algorithm: TensorCRO:{device}, Function: {test_function.__name__}:'
#                                  f'{test_function_instance.number_of_evaluations} eval, '
#                                  f'Dimension: {dimension}, Best fitness: {best_fit[0]},
#                                  Best individual: {best_ind[0]}')

def check_occurrence(this_simulation, stored_simulations):
    # We check if the current simulation is already done::
    for stored_simulation in stored_simulations:
        if this_simulation['function'] == stored_simulation['function'] and \
                this_simulation['dimension'] == stored_simulation['dimension'] and \
                this_simulation['algorithm'] == stored_simulation['algorithm'] and \
                this_simulation['seed'] == stored_simulation['seed']:
            return True
    return False


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    run_tests()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
