# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import time
import tensorflow as tf
import numpy as np
from .parameters import WindParameters, TurbineParameters


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def fitness_py(population: np.ndarray) -> np.ndarray:
    """
    Fitness function for the wind farm problem
    :param population: Individual to evaluate
    :type population: np.ndarray
    :return: Fitness values for the population
    """
    _aep = list()
    for turb_coords in population:
        #  Power produced by the wind farm from each wind direction
        pwr_produced = np.zeros_like(WindParameters.freq)
        # For each wind bin
        for idx, wind_dir_rad in enumerate(WindParameters.dirs_rad):
            # Find the farm's power for the current directionnp.recarray(frame_coords_in
            this_wind_speed = WindParameters.speed
            this_turb_diam = TurbineParameters.diam
            this_turb_ci = TurbineParameters.ci
            this_turb_co = TurbineParameters.co
            this_rated_ws = TurbineParameters.rated_ws
            this_rated_pwr = TurbineParameters.rated_pwr
            # Return the power produced by each turbine.
            # Shift coordinate frame of reference to downwind/crosswind
            # """Convert map coordinates to downwind/crosswind coordinates."""
            # Convert from meteorological polar system (CW, 0 deg.=N)
            # to standard polar system (CCW, 0 deg.=W)
            # Shift so North comes "along" x-axis, from left to right.
            # Constants to use below
            cos_dir = np.cos(-wind_dir_rad)
            sin_dir = np.sin(-wind_dir_rad)
            # Convert to downwind(x) & crosswind(y) coordinates
            # frame_coords = np.recarray(turb_coords.shape, np.dtype([('x', 'f8'), ('y', 'f8')]))
            frame_coords = np.zeros_like(turb_coords)
            frame_coords[0] = (turb_coords[0] * cos_dir) - (turb_coords[1] * sin_dir)
            frame_coords[1] = (turb_coords[0] * sin_dir) + (turb_coords[1] * cos_dir)
            # Use the Simplified Bastankhah Gaussian wake model for wake deficits
            # """Return each turbine's total loss due to wake from upstream turbines"""
            # Equations and values explained in <iea37-wakemodel.pdf>
            num_turb = len(frame_coords[0])
            # Array holding the wake deficit seen at each turbine
            loss = np.zeros(num_turb)
            for i in range(num_turb):  # Looking at each turb (Primary)
                loss_array = np.zeros(num_turb)  # Calculate the loss from all others
                for j in range(num_turb):  # Looking at all other turbs (Target)
                    x = frame_coords[0][i] - frame_coords[0][j]  # Calculate the x-dist
                    y = frame_coords[1][i] - frame_coords[1][j]  # And the y-offset
                    if x > 0.:  # If Primary is downwind of the Target
                        sigma = TurbineParameters.k * x + this_turb_diam / np.sqrt(8.)  # Calculate the wake loss
                        # Simplified Bastankhah Gaussian wake model
                        exponent = -0.5 * (y / sigma) ** 2
                        radical = 1. - TurbineParameters.ct / (8. * sigma ** 2 / this_turb_diam ** 2)
                        loss_array[j] = (1. - np.sqrt(radical)) * np.exp(exponent)
                    # Note that if the Target is upstream, loss is defaulted to zero
                # Total wake losses from all upstream turbs, using sqrt of sum of sqrs
                loss[i] = np.sqrt(np.sum(loss_array ** 2))
            # Effective windspeed is freestream multiplied by wake deficits
            wind_speed_eff = this_wind_speed * (1. - loss)
            # By default, the turbine's power output is zero
            turb_pwr = np.zeros(num_turb)
            # Check to see if turbine produces power for experienced wind speed
            for n in range(num_turb):
                # If we're between the cut-in and rated wind speeds
                if ((this_turb_ci <= wind_speed_eff[n])
                        and (wind_speed_eff[n] < this_rated_ws)):
                    # Calculate the curve's power
                    turb_pwr[n] = this_rated_pwr * ((wind_speed_eff[n] - this_turb_ci)
                                                    / (this_rated_ws - this_turb_ci)) ** 3
                # If we're between the rated and cut-out wind speeds
                elif ((this_rated_ws <= wind_speed_eff[n])
                      and (wind_speed_eff[n] < this_turb_co)):
                    # Produce the rated power
                    turb_pwr[n] = this_rated_pwr

            # Sum the power from all turbines for this direction
            pwr_produced[idx] = np.sum(turb_pwr)
        #  Convert power to AEP
        hrs_per_year = 365. * 24.
        aep = hrs_per_year * (WindParameters.freq * pwr_produced)
        aep /= 1.E6  # Convert to MWh
        _aep.append(aep)
    return np.array(_aep)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def fitness_py_np(population: np.ndarray) -> np.ndarray:
    """
    Fitness function for the wind farm problem
    :param population: Individual to evaluate
    :type population: np.ndarray
    :return: Fitness values for the population
    """
    this_wind_speed = WindParameters.speed
    this_turb_diam = TurbineParameters.diam
    this_turb_ci = TurbineParameters.ci
    this_turb_co = TurbineParameters.co
    this_rated_ws = TurbineParameters.rated_ws
    this_rated_pwr = TurbineParameters.rated_pwr
    hrs_per_year = 365. * 24. / 1.E6  # Convert to MWh
    aep = np.zeros((population.shape[0], WindParameters.dirs_rad.shape[-1]))

    for pop_idx, turb_coords in enumerate(population):
        #  Power produced by the wind farm from each wind direction
        pwr_produced = np.zeros_like(WindParameters.freq)
        # For each wind bin
        for idx, wind_dir_rad in enumerate(WindParameters.dirs_rad):
            # Compute sin and cos coords.
            cos_dir = np.cos(-wind_dir_rad)
            sin_dir = np.sin(-wind_dir_rad)
            # Convert to downwind(x) & crosswind(y) coordinates
            frame_coords = np.array([(turb_coords[0] * cos_dir) - (turb_coords[1] * sin_dir),
                                     (turb_coords[0] * sin_dir) + (turb_coords[1] * cos_dir)], dtype=np.float32)
            # Calculate the x and y distances between all turbines
            delta_x = frame_coords[0][:, np.newaxis] - frame_coords[0][np.newaxis, :]
            delta_y = frame_coords[1][:, np.newaxis] - frame_coords[1][np.newaxis, :]
            # Identify turbines that are downwind
            downwind_mask = delta_x > 0
            # Calculate the wake loss for downwind turbines
            sigma = TurbineParameters.k * delta_x + this_turb_diam / np.sqrt(8.)
            exponent = -0.5 * (delta_y / sigma) ** 2
            radical = 1. - TurbineParameters.ct / (8. * sigma ** 2 / this_turb_diam ** 2)
            loss_array = (1. - np.sqrt(radical)) * np.exp(exponent)
            # Set the loss to zero for upstream turbines
            loss_array[~downwind_mask] = 0
            # Calculate total wake losses from all upstream turbines
            loss = np.sqrt(np.sum(loss_array ** 2, axis=1))
            # Calculate effective windspeed
            wind_speed_eff = this_wind_speed * (1. - loss)

            # By default, the turbine's power output is zero
            turb_pwr = np.zeros(np.shape(frame_coords[-1]))

            # Check conditions for power production
            cut_in_mask = (this_turb_ci <= wind_speed_eff) & (wind_speed_eff < this_rated_ws)
            rated_mask = (this_rated_ws <= wind_speed_eff) & (wind_speed_eff < this_turb_co)

            # Calculate power output for turbines within cut-in and rated wind speeds
            turb_pwr[cut_in_mask] = this_rated_pwr * ((wind_speed_eff[cut_in_mask] - this_turb_ci) /
                                                      (this_rated_ws - this_turb_ci)) ** 3

            # Set power output to rated power for turbines within rated and cut-out wind speeds
            turb_pwr[rated_mask] = this_rated_pwr

            # Sum the power from all turbines for this direction
            pwr_produced[idx] = np.sum(turb_pwr)
        #  Convert power to AEP
        aep[pop_idx] = hrs_per_year * (WindParameters.freq * pwr_produced)
    return aep


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def fitness_np(population: np.ndarray) -> np.ndarray:
    """
    Fitness function for the wind farm problem
    :param population: Individual to evaluate
    :type population: np.ndarray
    :return: Fitness values for the population
    """
    this_wind_speed = WindParameters.speed
    this_turb_diam = TurbineParameters.diam
    this_turb_ci = TurbineParameters.ci
    this_turb_co = TurbineParameters.co
    this_rated_ws = TurbineParameters.rated_ws
    this_rated_pwr = TurbineParameters.rated_pwr
    hrs_per_year = 365. * 24. / 1.E6

    wind_dir_rad = WindParameters.dirs_rad

    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)

    tiled_x = np.tile(population[:, 0, np.newaxis], (1, population.shape[-1], 1)).transpose(0, 2, 1)
    tiled_y = np.tile(population[:, 1, np.newaxis], (1, population.shape[-1], 1)).transpose(0, 2, 1)

    _x = tiled_x * cos_dir - tiled_y * sin_dir
    _y = tiled_x * sin_dir + tiled_y * cos_dir

    delta_x = (_x[:, :, np.newaxis] - _x[:, np.newaxis, :]).transpose(0, 3, 1, 2)
    delta_y = (_y[:, :, np.newaxis] - _y[:, np.newaxis, :]).transpose(0, 3, 1, 2)

    downwind_mask = delta_x > 0

    sigma = TurbineParameters.k * delta_x + this_turb_diam / np.sqrt(8.)
    exponent = -0.5 * (delta_y / sigma) ** 2
    radical = 1. - TurbineParameters.ct / (8. * sigma ** 2 / this_turb_diam ** 2)
    loss_array = (1. - np.sqrt(radical)) * np.exp(exponent)
    loss_array[~downwind_mask] = 0
    loss = np.sqrt(np.sum(loss_array ** 2, axis=-1))

    wind_speed_eff = this_wind_speed * (1. - loss)

    turb_pwr = np.zeros_like(_x)

    cut_in_mask = (this_turb_ci <= wind_speed_eff) & (wind_speed_eff < this_rated_ws)
    rated_mask = (this_rated_ws <= wind_speed_eff) & (wind_speed_eff < this_turb_co)

    turb_pwr[cut_in_mask] = this_rated_pwr * ((wind_speed_eff[cut_in_mask] - this_turb_ci) /
                                              (this_rated_ws - this_turb_ci)) ** 3
    turb_pwr[rated_mask] = this_rated_pwr

    pwr_produced = np.sum(turb_pwr, axis=-1)
    aep = hrs_per_year * (WindParameters.freq * pwr_produced)
    return aep


@tf.function
def fitness_tf(population: tf.Tensor) -> tf.Tensor:
    """
    Fitness function for the wind farm problem
    :param population: Individual to evaluate
    :type population: np.ndarray
    :return: Fitness values for the population
    """
    population = tf.cast(population, dtype=tf.float32)
    this_wind_speed = tf.constant(WindParameters.speed, dtype=tf.float32)
    this_turb_diam = tf.constant(TurbineParameters.diam, dtype=tf.float32)
    this_turb_ci = tf.constant(TurbineParameters.ci, dtype=tf.float32)
    this_turb_co = tf.constant(TurbineParameters.co, dtype=tf.float32)
    this_rated_ws = tf.constant(TurbineParameters.rated_ws, dtype=tf.float32)
    this_rated_pwr = tf.constant(TurbineParameters.rated_pwr, dtype=tf.float32)
    hrs_per_year = tf.constant(365. * 24. / 1.E6, dtype=tf.float32)

    wind_dir_rad = tf.constant(WindParameters.dirs_rad, dtype=tf.float32)

    cos_dir = tf.cast(tf.cos(-wind_dir_rad), dtype=tf.float32)
    sin_dir = tf.cast(tf.sin(-wind_dir_rad), dtype=tf.float32)

    tiled_x = tf.tile(tf.expand_dims(population[:, 0], axis=-1), (1, 1, population.shape[-1]))
    tiled_y = tf.tile(tf.expand_dims(population[:, 1], axis=-1), (1, 1, population.shape[-1]))

    tiled_cos_dir = tf.tile(tf.expand_dims(tf.expand_dims(cos_dir, axis=0), axis=-1),
                            (population.shape[0], 1, cos_dir.shape[-1]))
    tiled_sin_dir = tf.tile(tf.expand_dims(tf.expand_dims(sin_dir, axis=0), axis=-1),
                            (population.shape[0], 1, sin_dir.shape[-1]))

    tiled_sin_dir = tf.transpose(tiled_sin_dir, perm=[0, 2, 1])
    tiled_cos_dir = tf.transpose(tiled_cos_dir, perm=[0, 2, 1])

    _x = tiled_x * tiled_cos_dir - tiled_y * tiled_sin_dir
    _y = tiled_x * tiled_sin_dir + tiled_y * tiled_cos_dir

    delta_x = tf.transpose(_x[:, tf.newaxis, :] - _x[:, :, tf.newaxis], perm=(0, 3, 2, 1))
    delta_y = tf.transpose(_y[:, tf.newaxis, :] - _y[:, :, tf.newaxis], perm=(0, 3, 2, 1))

    downwind_mask = delta_x > 0

    sigma = TurbineParameters.k * delta_x + this_turb_diam / tf.sqrt(8.0)
    exponent = -0.5 * (delta_y / sigma) ** 2
    radical = 1.0 - TurbineParameters.ct / (8.0 * sigma ** 2 / this_turb_diam ** 2)
    loss_array = (1.0 - tf.sqrt(radical)) * tf.exp(exponent)
    loss_array = tf.where(downwind_mask, loss_array, tf.zeros_like(loss_array))
    loss = tf.sqrt(tf.reduce_sum(loss_array ** 2, axis=-1))

    wind_speed_eff = this_wind_speed * (1.0 - loss)

    turb_pwr = tf.zeros_like(_x)

    cut_in_mask = (this_turb_ci <= wind_speed_eff) & (wind_speed_eff < this_rated_ws)
    rated_mask = (this_rated_ws <= wind_speed_eff) & (wind_speed_eff < this_turb_co)

    turb_pwr = tf.where(cut_in_mask, this_rated_pwr * ((wind_speed_eff - this_turb_ci) / (this_rated_ws - this_turb_ci)) ** 3, turb_pwr)
    turb_pwr = tf.where(rated_mask, this_rated_pwr, turb_pwr)

    pwr_produced = tf.reduce_sum(turb_pwr, axis=-1)
    aep = hrs_per_year * (WindParameters.freq * pwr_produced)
    return aep
#
#
# if __name__ == "__main__":
#     # Calculate the AEP from ripped values
#     x = -np.array([- 168.22,
#       - -549.82,
#       - -57.405,
#       - -894.98,
#       - -78.127,
#       - 1013.3,
#       - 329.28,
#       - -459.74,
#       - -312.11,
#       - 566.65,
#       - 932.51,
#       - -1214.4,
#       - -672.93,
#       - 790.69,
#       - 462.17,
#       - -19.596])
#     y = -np.array([- 72.345,
#       - -171.07,
#       - -283.3,
#       - 489.09,
#       - -1133.7,
#       - 722.97,
#       - 311.2,
#       - 433.87,
#       - -667.56,
#       - 175.67,
#       - 226.99,
#       - -102.21,
#       - -785.96,
#       - -91.841,
#       - -1139.7,
#       - 312.56])
#     np.round(fitness_tf(tf.convert_to_tensor([np.array([x, y])])) * 1.E6, 3)
#     tik_0 = time.perf_counter()
#     v_0 = (np.round(fitness_py(np.array([np.array([x, y])])), 3))
#     tok_0 = time.perf_counter()
#     v_1 = (np.round(fitness_np(np.array([np.array([x, y])])), 3))
#     tok_1 = time.perf_counter()
#     v_2 = (np.round(fitness_py_np(np.array([np.array([x, y])])), 3))
#     tok_2 = time.perf_counter()
#     v_3 = (np.round(fitness_tf(tf.convert_to_tensor([np.array([x, y])])), 3))
#     tok_3 = time.perf_counter()
#     print(f'fitness_py: {tok_0 - tik_0}')
#     print(f'fitness_np: {tok_1 - tok_0}')
#     print(f'fitness_py_np: {tok_2 - tok_1}')
#     print(f'fitness_tf: {tok_3 - tok_2}')
#
#     print('Error on fitness_np: ', np.sum(np.abs(v_0 - v_1)))
#     print('Error on fitness_pn: ', np.sum(np.abs(v_0 - v_2)))
#     print('Error on fitness_tf: ', np.sum(np.abs(v_0 - v_3)))
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
