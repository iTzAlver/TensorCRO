# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import tensorflow as tf
from src.tensorcro import UniformCrossover, GaussianCrossover, MultipointCrossover, HarmonySearch, \
    RandomSearch, ComposedSubstrate, Mutation, MaskedCrossover, BLXAlphaCrossover, DifferentialSearch, TensorCro


def fitness_function(individuals: tf.RaggedTensor):
    __retval = tf.reduce_sum(individuals, axis=-1)
    return __retval


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main() -> None:
    reef_shape = (5, 4)
    initializer = tf.zeros((*reef_shape, 5), dtype=tf.float32)
    fitness = tf.zeros(reef_shape, dtype=tf.float32)
    directives = tf.convert_to_tensor([(-1, -0.5, 0, 0.5, 0), (0, 1, 1, 1, 0.5)], dtype=tf.float32)

    substrates = [
        UniformCrossover(),
        GaussianCrossover(),
        MultipointCrossover([3]),
        HarmonySearch(directives),
        Mutation('gaussian'),
        MaskedCrossover([0, 0, 1, 1, 0]),
        BLXAlphaCrossover(directives),
        DifferentialSearch(directives),
        ComposedSubstrate(
            UniformCrossover(),
            Mutation('uniform'),
            name='GeneticStrategy'
        )
    ]
    common_substrate = RandomSearch(directives)

    for substrate in substrates:
        print(f"Testing {substrate.__repr__()} substrate")
        cro = TensorCro(reef_shape=reef_shape, subs=[common_substrate, substrate])
        reef = cro.fit(fitness_function, directives, init=(initializer, fitness), seed=0, shards=1)
        cro.watch_replay(lock=True)
        print(f"Best solution: {reef[0]} vs {directives[1].numpy().tolist()}")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
