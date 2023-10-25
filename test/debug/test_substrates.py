# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from tensorcro import TensorCro, ParticleSwarmOptimization, SimulatedAnnealing
import tensorflow as tf
import logging
import time
logging.basicConfig(level=logging.INFO)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@tf.function
def fitness_function(individuals: tf.RaggedTensor):
    __retval = tf.reduce_sum(individuals, axis=-1)
    return __retval


def main() -> None:
    logging.info('[+] Connected to substrate test...')
    # Create substrate:
    directives = tf.convert_to_tensor([(0., 0., 0., 0.5, 0.), (1., 1., 1., 1., 0.5)], dtype_hint=tf.float32)
    reef_shape = (10, 20)

    sub_shape = (10, 10)  # Pre-compute.
    pso = ParticleSwarmOptimization(directives, shape=sub_shape, inertia=0.5, cognition=1., social=1.)
    simulated_annealing = SimulatedAnnealing(directives, kmax=100, bandwidth=0.2, shape=sub_shape)

    subs = [pso, simulated_annealing]
    t_cro = TensorCro(reef_shape, subs=subs)
    tik = time.perf_counter()
    t_cro.fit(fitness_function, directives, max_iter=5_000, device='/GPU:0', seed=0, shards=500)
    tak = time.perf_counter()
    print(f"GPU time: {tak - tik}")
    t_cro.save_replay('./replay')
    t_cro.watch_replay('./replay', mp=False)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
