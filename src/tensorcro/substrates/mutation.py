# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import tensorflow as tf
from .substrate import CROSubstrate
FUNC = {
    'uniform': tf.random.uniform,
    'gaussian': tf.random.normal,
    'truncated_normal': tf.random.truncated_normal,
    'poisson': tf.random.poisson,
    'gamma': tf.random.gamma
}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Mutation(CROSubstrate):
    def __init__(self, mutation_type: str, **kwargs):
        """
        Mutation class. It is a CROSubstrate that applies a mutation to the individuals. The supported mutation types
        are:
            - uniform (uniform): tf.random.uniform
            - gaussian (normal): tf.random.normal
            - truncated_normal (truncated normal): tf.random.truncated_normal
            - poisson (poisson): tf.random.poisson
            - gamma (gamma): tf.random.gamma
        :param mutation_type: Mutation type as a string.
        :param kwargs: Arguments for the mutation function.
        """
        if mutation_type not in FUNC:
            raise ValueError(f"TensorCRO:Mutation: Mutation type {mutation_type} not supported."
                             f"\nSupported types: {list(FUNC.keys())}")
        self.func = FUNC[mutation_type]
        self.arguments = kwargs

    def _call(self, individuals: tf.Tensor, **kwargs) -> tf.Tensor:
        mutation = self.func(tf.shape(individuals), dtype=tf.float32, **self.arguments)
        return tf.math.add(individuals, mutation)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
