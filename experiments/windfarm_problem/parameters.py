# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from dataclasses import dataclass
import numpy as np


@dataclass
class TurbineParameters:
    """
    Turbine static parameters
    """
    diam: np.float32 = np.float32(2 * 65.0)
    ci: np.float32 = np.float32(4.0)
    co: np.float32 = np.float32(25.0)
    min_dist: np.float32 = np.float32(2 * diam)

    rated_ws: np.float32 = np.float32(9.8)
    rated_pwr: np.float32 = np.float32(3_350_000.0)

    ct: np.float32 = np.float32(4.0 * 1. / 3. * (1.0 - 1. / 3.))
    k: np.float32 = np.float32(0.0324555)


@dataclass
class WindParameters:
    """
    Wind static parameters
    """
    freq: np.ndarray = np.array([.025,  .024,  .029,  .036, .063,  .065,  .100,  .122, .063,
                                 .038,  .039,  .083, .213,  .046,  .032,  .022], np.float32)
    speed: np.float32 = np.float32(9.8)
    dirs_deg: np.ndarray = np.array([0., 22.5, 45., 67.5, 90., 112.5, 135., 157.5, 180., 202.5, 225.,
                                     247.5, 270., 292.5, 315., 337.5], np.float32)
    dirs_rad: np.ndarray = np.radians(270. - dirs_deg)
    park_diam: np.ndarray = np.array([1_300, 2_000, 3_000], np.int32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
