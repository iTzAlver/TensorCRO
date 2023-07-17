# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import json
import numpy as np
import logging
# Algorithms:
from tensorcro import TensorCro, UniformCrossover, MultipointCrossover, HarmonySearch, RandomSearch, \
    ComposedSubstrate, Mutation
from other_algorithms import GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm
# Many local minima functions:
from test_functions import AckleyFunction, BukinFunction6, CrossInTrayFunction, DropWaveFunction, EggHolderFunction, \
        GramacyLeeFunction, GrieWankFunction, HolderTableFunction, RastriginFunction, LevyFunction, ShubertFunction
# Bowl shape functions:
from test_functions import PermZDB, HyperEpsiloid, HyperSphere
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
SEED = 2023
FUNCTIONS = {'local_minima': [
    AckleyFunction, BukinFunction6, CrossInTrayFunction, DropWaveFunction, EggHolderFunction,
    GramacyLeeFunction, GrieWankFunction, HolderTableFunction, RastriginFunction, LevyFunction,
    ShubertFunction],
    'bowl_shape':
        [PermZDB, HyperEpsiloid, HyperSphere],
    'valley_shape':
        [DixonPrice, Rosenbrock],
    'flat_shape':
        [Easom, Michalewicz],
    'other':
        [StyblinskiTang, Powell]}
ALGORITHMS = [TensorCro, GeneticAlgorithm, PSOAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm]
DIMENSIONS = [5, 50, 100, 150, 200, 250, 300]
RESULTS_PATH = './results/'


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
    # - Seed number:            2023
    # - Number of iterations:   Time limited (2 - 5 - 10) minutes.
    # - Number of individuals:  Dimension dependent: 20 times the dimension.
    # - Number of dimensions:   Variable.
    # - Function:               All implemented functions.
    # - Algorithms:             All implemented algorithms.
    for times in [2, 5, 10]:
        for algorithm in ALGORITHMS:
            for function_type, function_list in FUNCTIONS.items():
                for test_function in function_list:
                    for dimension in DIMENSIONS:
                        # We compose the current test:
                        test_function_instance = test_function()
                        core_bounds = test_function_instance.bounds
                        bounds = np.array([core_bounds] * dimension, dtype=np.float32).T
                        # Run the algorithm:
                        if algorithm is not SimulatedAnnealingAlgorithm and algorithm is not TensorCro:
                            ai = algorithm(20 * dimension)
                        elif algorithm is TensorCro:
                            uniform_crossover = UniformCrossover()
                            harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=bounds)
                            random_search = RandomSearch(bounds, 0.2)
                            genetic_algorithm = ComposedSubstrate(
                                MultipointCrossover([dimension // 2]),
                                Mutation('gaussian', mean=0.0, stddev=0.05), name='GeneticAlgorithm'
                            )
                            only_mutation = ComposedSubstrate(
                                Mutation('gaussian', mean=0.0, stddev=0.1), name='OnlyMutation'
                            )
                            subs = [uniform_crossover, harmony_search, random_search, genetic_algorithm, only_mutation]
                            ai = TensorCro(reef_shape=(5, dimension * 4), subs=subs)
                        else:
                            ai = algorithm()
                        # We run the algorithm:
                        best_ind, best_fit = ai.fit(test_function_instance, bounds, int(1e10),
                                                    time_limit=times * 60, seed=SEED)
                        # We convert the results to list if they are not:
                        if not isinstance(best_ind, np.ndarray):
                            best_ind = best_ind.numpy().tolist()
                            best_fit = best_fit.numpy().tolist()
                        # Save the results:
                        with open(RESULTS_PATH + 'test_1.json', 'w') as f:
                            json.dump({'algorithm': algorithm.__name__,
                                       'function': test_function.__name__,
                                       'dimension': dimension,
                                       'best_fit': float(best_fit[0]),
                                       'best_ind': list(best_ind[0]),
                                       'time_limit': times * 60,
                                       'num_eval': test_function_instance.number_of_evaluations}, f, indent=4)
                            f.write('\n')
                        # Logging:
                        logging.info(f'Test 1: Algorithm: {algorithm.__name__}, Function: {test_function.__name__}:'
                                     f'{test_function_instance.number_of_evaluations} eval, '
                                     f'Dimension: {dimension}, Best fitness: {best_fit[0]}, Best individual: '
                                     f'{best_ind[0]}')


def test_2() -> None:
    # We run the test 2 with the following parameters:
    # - Seed number:            2023
    # - Number of iterations:   10_000 fitness evaluations.
    # - Number of individuals:  Dimension dependent: 20 times the dimension.
    # - Number of dimensions:   Variable.
    # - Function:               All implemented functions.
    # - Algorithms:             All implemented algorithms.
    for algorithm in ALGORITHMS:
        for function_type, function_list in FUNCTIONS.items():
            for test_function in function_list:
                for dimension in DIMENSIONS:
                    # Run the algorithm:
                    if algorithm is not SimulatedAnnealingAlgorithm:
                        ai = algorithm(20 * dimension)
                    else:
                        ai = algorithm()
                    # We run the algorithm:
                    test_function_instance = test_function()
                    core_bounds = test_function_instance.bounds
                    bounds = np.array([core_bounds] * dimension).T
                    best_ind, best_fit = ai.fit(test_function_instance, bounds, -1, seed=SEED)
                    # Save the results:
                    with open(RESULTS_PATH + 'test_2.json', 'w') as f:
                        json.dump({'algorithm': algorithm.__name__,
                                   'function': test_function.__name__,
                                   'dimension': dimension,
                                   'best_fit': float(best_fit),
                                   'best_ind': list(best_ind),
                                   'num_eval': test_function_instance.number_of_evaluations}, f, indent=4)
                        f.write('\n')
                    # Logging:
                    logging.info(f'Test 2: Algorithm: {algorithm.__name__}, Function: {test_function.__name__}:'
                                 f'{test_function_instance.number_of_evaluations} eval, '
                                 f'Dimension: {dimension}, Best fitness: {best_fit[0]}, Best individual: {best_ind[0]}')


def test_3() -> None:
    # We run the test 2 with the following parameters:
    # - Seed number:            2023
    # - Number of iterations:   10 minutes and 50k fitness evaluations per function.
    # - Number of individuals:  Dimension dependent: 20 times the dimension.
    # - Number of dimensions:   Variable.
    # - Function:               All implemented functions.
    # - Algorithms:             TensorCRO(GPU), CRO(CPU).
    for device in ['/GPU:0', '/CPU:0']:
        for function_type, function_list in FUNCTIONS.items():
            for test_function in function_list:
                for dimension in DIMENSIONS:
                    # Run the algorithm:
                    ai = TensorCro()
                    # We run the algorithm:
                    test_function_instance = test_function()
                    core_bounds = test_function_instance.bounds
                    bounds = np.array([core_bounds] * dimension).T
                    best_ind, best_fit = ai.fit(test_function_instance, bounds, -1, seed=SEED, device=device)
                    # Save the results:
                    with open(RESULTS_PATH + 'test_3.json', 'w') as f:
                        json.dump({'algorithm': f'TensorCRO:{device}',
                                   'function': test_function.__name__,
                                   'dimension': dimension,
                                   'best_fit': float(best_fit),
                                   'best_ind': list(best_ind),
                                   'num_eval': test_function_instance.number_of_evaluations}, f, indent=4)
                        f.write('\n')
                    # Logging:
                    logging.info(f'Test 3: Algorithm: TensorCRO:{device}, Function: {test_function.__name__}:'
                                 f'{test_function_instance.number_of_evaluations} eval, '
                                 f'Dimension: {dimension}, Best fitness: {best_fit[0]}, Best individual: {best_ind[0]}')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    run_tests()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
