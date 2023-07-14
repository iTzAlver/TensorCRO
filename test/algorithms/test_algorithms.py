# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
"""
File info:
"""
# Import statements:
import numpy as np
import time
from experiments.other_algorithms import GeneticAlgorithm, HarmonySearchAlgorithm, SimulatedAnnealingAlgorithm, \
    PSOAlgorithm, FireflyAlgorithm
from test_functions import AckleyFunction


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def test(dimension):
    tik = time.perf_counter()
    sa = test_sa(dimension=dimension)
    tok = time.perf_counter()
    ga = test_ga(dimension=dimension)
    tok2 = time.perf_counter()
    hs = test_hs(dimension=dimension)
    tok3 = time.perf_counter()
    ps = test_ps(dimension=dimension)
    tok4 = time.perf_counter()
    ff = test_ff(dimension=dimension)
    tok5 = time.perf_counter()
    print(f"\n")
    print(f"Best solution in GA: {ga:.4f} seed 1998 in {tok2 - tok:.4f} seconds")
    print(f"Best solution in HS: {hs:.4f} seed 1998 in {tok3 - tok2:.4f} seconds")
    print(f"Best solution in SA: {sa:.4f} seed 1998 in {tok - tik:.4f} seconds")
    print(f"Best solution in PS: {ps:.4f} seed 1998 in {tok4 - tok3:.4f} seconds")
    print(f"Best solution in FF: {ff:.4f} seed 1998 in {tok5 - tok4:.4f} seconds")


def test_ga(dimension):
    fitness = AckleyFunction()
    minimum, maximum = fitness.bounds
    bounds = np.array([(minimum, maximum) for _ in range(dimension)]).T
    ga = GeneticAlgorithm(individuals=6 * dimension, mutation_rate=0.6, crossover_rate=0.5, mutation_variance=0.1)
    _, best_fit = ga.fit(fitness, bounds, 10_000, verbose=True, seed=1998)
    return best_fit[0]


def test_hs(dimension):
    fitness = AckleyFunction()
    minimum, maximum = fitness.bounds
    bounds = np.array([(minimum, maximum) for _ in range(dimension)]).T
    ga = HarmonySearchAlgorithm(individuals=6 * dimension, hmcr=0.9, par=0.2, bw=0.1)
    _, best_fit = ga.fit(fitness, bounds, 10_000, verbose=True, seed=1998)
    return best_fit[0]


def test_sa(dimension):
    fitness = AckleyFunction()
    minimum, maximum = fitness.bounds
    bounds = np.array([(minimum, maximum) for _ in range(dimension)]).T
    ga = SimulatedAnnealingAlgorithm()
    _, best_fit = ga.fit(fitness, bounds, 10_000, verbose=True, seed=1998)
    return best_fit[0]


def test_ps(dimension):
    fitness = AckleyFunction()
    minimum, maximum = fitness.bounds
    bounds = np.array([(minimum, maximum) for _ in range(dimension)]).T
    ga = PSOAlgorithm(6 * dimension)
    _, best_fit = ga.fit(fitness, bounds, 10_000, verbose=True, seed=1998)
    return best_fit[0]


def test_ff(dimension):
    fitness = AckleyFunction()
    minimum, maximum = fitness.bounds
    bounds = np.array([(minimum, maximum) for _ in range(dimension)]).T
    ga = FireflyAlgorithm(6 * dimension)
    _, best_fit = ga.fit(fitness, bounds, 10_000, verbose=True, seed=1998)
    return best_fit[0]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    test(10)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
