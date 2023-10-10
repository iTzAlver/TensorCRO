# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import time
import numpy as np
import tensorflow as tf
from src.tensorcro import TensorCro
from src.tensorcro import UniformCrossover, GaussianCrossover, MultipointCrossover, HarmonySearch, \
    RandomSearch, ComposedSubstrate, Mutation, MaskedCrossover, BLXAlphaCrossover, DifferentialSearch
from src.tensorcro.substrates.substrate import CROSubstrate


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@tf.function
def fitness_function(individuals: tf.RaggedTensor):
    __retval = tf.reduce_sum(individuals, axis=-1)
    return __retval


def main() -> None:
    od = [(0., 0., 0., 0.5, 0.), (1., 1., 1., 1., 0.5)]
    directives = tf.convert_to_tensor([(0., 0., 0., 0.5, 0.), (1., 1., 1., 1., 0.5)], dtype_hint=tf.float32)
    reef_shape = (10, 20)

    uniform_crossover = UniformCrossover()
    gaussian_crossover = GaussianCrossover(mean=0.5, stddev=1.0)
    multipoint_crossover = MultipointCrossover([3])
    harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=directives)
    random_search = RandomSearch(directives, 0.2)
    genetic_algorithm = ComposedSubstrate(
        MultipointCrossover([3]),
        Mutation('gaussian', stddev=0.001),
        Mutation('uniform', minval=directives[0], maxval=directives[1]))
    masked_crossover = MaskedCrossover([0, 0, 0, 1, 1])
    blxalpha_crossover = BLXAlphaCrossover(directives, scale=0.5)
    differential_evolution = DifferentialSearch(directives, 0.8, 0.7)
    random_search_2 = RandomSearch(directives, 10)

    subs = [uniform_crossover, gaussian_crossover, multipoint_crossover, harmony_search, random_search,
            genetic_algorithm, masked_crossover, blxalpha_crossover, differential_evolution, random_search_2]

    test_utils(directives, subs)
    t_cro = TensorCro(reef_shape, subs=subs)
    t_cro.fit(fitness_function, od, max_iter=10, device='/CPU:0', seed=0, shards=1)  # Warm up
    tik = time.perf_counter()
    t_cro.fit(fitness_function, directives, max_iter=10, device='/CPU:0', seed=0, shards=1)
    tok = time.perf_counter()
    t_cro.fit(fitness_function, directives, max_iter=10, device='/GPU:0', seed=0, shards=1)
    tak = time.perf_counter()
    t_cro.fit(fitness_function, directives, max_iter=int(1e10), device='/GPU:0', seed=0, shards=1, time_limit=60)
    print(f"CPU time: {tok - tik}")
    print(f"GPU time: {tak - tok}")
    print(f'GPU speed up over CPU: {(tok - tik) / (tak - tok)}')
    t_cro.save_replay('./replay')
    t_cro.watch_replay('./replay', mp=False)
    t_cro.watch_replay('./replay', lock=True)


def test_utils(directives, subs):
    try:
        TensorCro((10, 19), subs=subs)
        raise AssertionError('TensorCro should raise an exception when reef_shape[1] is not divisible by 2 substrates')
    except ValueError:
        zeros = tf.zeros((len(subs), 10, 2, directives.shape[-1]))
        fitness = tf.zeros((10, 2 * len(subs)))
        tc = TensorCro((10, 2 * len(subs)), subs=subs)
        tc.fit(fitness_function, directives, max_iter=5, init=(zeros, fitness))
        print(tc)
        dummy_substrate = CROSubstrate()
        assert np.allclose(zeros.numpy(), dummy_substrate(zeros).numpy()), \
            'CROSubstrate should return the same individuals, as a dummy substrate.'
    try:
        Mutation('non_existing_mutation')
        raise AssertionError('Mutation should raise an exception when mutation_type is not valid')
    except ValueError:
        pass


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
