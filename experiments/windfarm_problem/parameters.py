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


# @dataclass
# class WindParameters36:
#     """
#     Wind static parameters
#     """
#     freq: np.ndarray = np.array([.025,  .024,  .029,  .036, .063,  .065,  .100,  .122, .063,
#                                  .038,  .039,  .083, .213,  .046,  .032,  .022], np.float32)
#     speed: np.float32 = np.float32(9.8)
#     dirs_deg: np.ndarray = np.array([0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.,
#                                     110., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210.,
#                                     220., 230., 240., 250., 260., 270., 280., 290., 300., 310., 320.,
#                                     330., 340., 350.], np.float32)
#     dirs_rad: np.ndarray = np.radians(270. - dirs_deg)
#     park_diam: np.ndarray = np.int32(2_000, np.int32)
#
#
# @dataclass
# class WindParameters64:
#     """
#     Wind static parameters
#     """
#     freq: np.ndarray = np.array([.025,  .024,  .029,  .036, .063,  .065,  .100,  .122, .063,
#                                  .038,  .039,  .083, .213,  .046,  .032,  .022], np.float32)
#     speed: np.float32 = np.float32(9.8)
#     dirs_deg: np.ndarray = np.array([0., 5.625, 11.25, 16.875,  22.5,  28.125,  33.75,
#                                     39.375,  45.,  50.625,  56.25,  61.875,  67.5,  73.125,
#                                     78.75,  84.375,  90.,  95.625, 101.25, 106.875, 112.5,
#                                     118.125, 123.75, 129.375, 135., 140.625, 146.25, 151.875,
#                                     157.5, 163.125, 168.75, 174.375, 180., 185.625, 191.25,
#                                     196.875, 202.5, 208.125, 213.75, 219.375, 225., 230.625,
#                                     236.25, 241.875, 247.5, 253.125, 258.75, 264.375, 270.,
#                                     275.625, 281.25, 286.875, 292.5, 298.125, 303.75, 309.375,
#                                     315., 320.625, 326.25, 331.875, 337.5, 343.125, 348.75,
#                                     354.375], np.float32)
#     dirs_rad: np.ndarray = np.radians(270. - dirs_deg)
#     park_diam: np.ndarray = np.int32(3_000, np.int32)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
