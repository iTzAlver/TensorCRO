# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import time
import tensorflow as tf


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
@tf.function
def fitness_function(a, b):
    tf.multiply(a, b)
    tf.multiply(b, a)


def main() -> None:
    shape = 22000
    a = tf.random.uniform(shape=(shape, shape), minval=0, maxval=1, dtype=tf.float32)
    b = tf.random.uniform(shape=(shape, shape), minval=0, maxval=1, dtype=tf.float32)
    tik = time.perf_counter()
    with tf.device('/CPU:0'):
        tf.multiply(a, b)
        tf.multiply(b, a)
    tok = time.perf_counter()
    with tf.device('/GPU:0'):
        tf.multiply(a, b)
        tf.multiply(b, a)
    tak = time.perf_counter()
    with tf.device('/CPU:0'):
        fitness_function(a, b)
    tek = time.perf_counter()
    with tf.device('/GPU:0'):
        fitness_function(a, b)
    tuk = time.perf_counter()
    print(f'CPU: {tok - tik} seconds')
    print(f'GPU: {tak - tok} seconds')
    print(f'CPU (tfunc): {tek - tak} seconds')
    print(f'GPU (tfunc): {tuk - tek} seconds')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
