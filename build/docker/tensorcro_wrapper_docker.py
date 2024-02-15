# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import json
import os
import importlib.util
import tensorcro
import tensorflow as tf


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def docker_run(base_path: str = './') -> None:
    """
    This function is called by the docker container to run the optimization process.
    :param base_path: Base path of the project.
    :return: None
    """
    # Config logging:
    logging.basicConfig(level=logging.INFO)
    # Set handler to logging path:
    logging.basicConfig(filename=os.path.join(base_path, 'logging.log'), filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    # Logging:
    logging.info("[+] Connected to TensorCRO.")

    # Load config json:
    __base_config = os.path.join(base_path, 'config.json')
    logging.info(f"[i] Loading config file: {__base_config}")
    with open(__base_config, 'r') as file:
        config = json.load(file)

    # Check if the fitness file is present:
    fitness_file = os.path.join(base_path, 'fitness.py')
    if not os.path.exists(fitness_file):
        logging.error(f"[x] Fitness file not found: {fitness_file}")
        raise FileNotFoundError(f"Fitness file not found: {fitness_file}")
    else:
        spec = importlib.util.spec_from_file_location("fitness", fitness_file)
        fitness_function = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fitness_function)
    logging.info(f"[i] Fitness file loaded: {fitness_file}")

    # Gather main trees:
    sim_name = config['name']
    sim_params = config['parameters']
    sim_dims = config['dims']
    sim_subs = config['substrates']
    sim_fit = config['fit']

    # Build the directives vector:
    for sub in








# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    docker_run()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
