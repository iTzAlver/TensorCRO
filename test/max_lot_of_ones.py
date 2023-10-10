# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import logging
import time
import tensorflow as tf
from src.tensorcro import TensorCro
from src.tensorcro import UniformCrossover, MultipointCrossover, HarmonySearch, \
    RandomSearch, ComposedSubstrate, Mutation


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@tf.function
def fitness_function(individuals: tf.RaggedTensor):
    __retval = tf.reduce_sum(individuals, axis=-1)
    return __retval


def main() -> None:
    n_ones = 1_000_000
    directives = tf.convert_to_tensor([[0] * n_ones, [1] * n_ones], dtype_hint=tf.float32)
    reef_shape = (10, 20)

    uniform_crossover = UniformCrossover()
    harmony_search = HarmonySearch(hmc_r=0.8, pa_r=0.1, bandwidth=0.05, directives=directives)
    random_search = RandomSearch(directives, 0.2)
    genetic_algorithm = ComposedSubstrate(
        MultipointCrossover([n_ones // 2]),
        Mutation('gaussian', mean=0.0, stddev=0.05), name='GeneticAlgorithm'
    )

    subs = [uniform_crossover, harmony_search, random_search, genetic_algorithm]

    t_cro = TensorCro(reef_shape, subs=subs)
    # Warm up.
    t_cro.fit(fitness_function, directives, max_iter=5, device='/CPU:0', seed=0, shards=5, save=False)
    # CPU speed.
    tik = time.perf_counter()
    t_cro.fit(fitness_function, directives, max_iter=200, device='/CPU:0', seed=0, shards=5, save=False)
    logging.warning('CPU finished')
    # GPU speed.
    tok = time.perf_counter()
    best = t_cro.fit(fitness_function, directives, max_iter=200, device='/GPU:0', seed=0, shards=5, save=False)[0]
    logging.warning('GPU finished')
    # Print results.
    tak = time.perf_counter()
    print(f'GPU speed up over CPU: {(tok - tik) / (tak - tok)}')
    print(f'Best individual: {best}: {fitness_function(tf.convert_to_tensor([best]))}')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
