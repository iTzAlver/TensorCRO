# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from .slcro import TensorCro, TF_INF
from .__special__ import __version__, __replay_path__, __src_path__
from .replay import watch_replay, SlackCallback, LightCallback
from .substrates import UniformCrossover, GaussianCrossover, MaskedCrossover, MultipointCrossover, BLXAlphaCrossover, \
    ComposedSubstrate, Mutation, ParticleSwarmOptimization, SimulatedAnnealing, EstimationDistribution, \
    EnergyReduction, PieceLossGain, CROSubstrate, HarmonyMutation, CoordinateDescent
from .substrates import PermutationCrossover as ExperimentalPermutationCrossover
from .substrates import HarmonySearch, RandomSearch, DifferentialSearch
# Format file:
import os
if not os.path.exists(__src_path__ + '/tmp'):
    os.mkdir(__src_path__ + '/tmp')
if not os.path.exists(__replay_path__):
    os.mkdir(__replay_path__)
if not os.path.exists(__replay_path__ + '/replay'):
    os.mkdir(__replay_path__ + '/replay')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
