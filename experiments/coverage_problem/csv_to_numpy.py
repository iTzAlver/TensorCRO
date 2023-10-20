# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
import pandas as pd


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def csv_to_numpy(map_path: str, device_path: str) -> tuple:
    """
    This function reads a csv file and returns a numpy array.
    :param map_path: Path to the map csv file.
    :param device_path: Path to the device csv file.
    :return: Tuple of numpy arrays: [0] -> Map, [1] -> Devices.
    """
    try:
        # Load csv file:
        df = pd.read_csv(map_path, header=0, delimiter=';')
        # Convert to numpy array:
        np_array = df.to_numpy()
        # Load device csv file:
        df_dev = pd.read_csv(device_path, header=0, delimiter=';')
        # Drop 'Nombre' column:
        df_dev = df_dev.drop(['Nombre', 'Angulo (grados)', 'Identificador'], axis=1)
        # Convert to numpy array:
        np_array_dev = df_dev.to_numpy()
    except Exception as e:
        logging.error("Error reading csv file: {}".format(e))
        raise ValueError("Error reading csv file: {}".format(e))
    return np_array, np_array_dev


def format_array(array_map: np.ndarray, device_array: np.ndarray) -> tuple:
    """
    This function formats the map array.
    :param array_map: Map array.
    :param device_array: Device array.
    :return: Tuple of numpy arrays: [0] -> Coordinates, [1] -> Is position, [2] -> Coverage boolean.
    """
    coordinates = array_map[:, 0:2]
    is_position = array_map[:, 2]
    coverage_boolean = array_map[:, 3:]
    distance = device_array[:, 0]
    cost = device_array[:, 1]
    return (np.array(coordinates, dtype=np.float32),
            np.array(is_position, np.int16),
            np.array(coverage_boolean, np.int16),
            np.array(distance, np.float32),
            np.array(cost, np.int16))
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
