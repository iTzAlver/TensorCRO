# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# https://github.com/eliben/deep-learning-samples/blob/master/tensorflow-samples/conv2d-numpy.py
def conv2d_single_channel(_input, w):
    """Two-dimensional convolution of a single channel.

    Uses SAME padding with 0s, a stride of 1 and no dilation.

    input: input array with shape (height, width)
    w: filter array with shape (fd, fd) with odd fd.

    Returns a result with the same shape as input.
    """
    assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

    # SAME padding with zeros: creating a new padded array to simplify index
    # calculations and to avoid checking boundary conditions in the inner loop.
    # padded_input is like input, but padded on all sides with
    # half-the-filter-width of zeros.
    npad = [(0, 0), (0, 0), (0, 0)]
    npad[1:] = [(w.shape[0] // 2, w.shape[0] // 2), (w.shape[0] // 2, w.shape[0] // 2)]
    padded_input = np.pad(_input, pad_width=tuple(npad), mode='constant', constant_values=0)

    output = np.zeros_like(_input)
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            # This inner double loop computes every output element, by
            # multiplying the corresponding window into the input with the
            # filter.
            for fi in range(w.shape[0]):
                for fj in range(w.shape[1]):
                    output[:, i, j] += padded_input[:, i + fi, j + fj] * w[fi, fj]
    return np.clip(output, 0, 1).astype(np.int32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
