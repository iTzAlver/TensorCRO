# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from .slcro import TensorCro
from .__special__ import __version__
from .replay import watch_replay
from .substrates import UniformCrossover, GaussianCrossover, MaskedCrossover, MultipointCrossover, BLXAlphaCrossover, \
    ComposedSubstrate, Mutation
from .substrates import HarmonySearch, RandomSearch, DifferentialSearch
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
