# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
import time
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
        fitness_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fitness_module)
        fitness_function = fitness_module.fitness
    logging.info(f"[i] Fitness file loaded: {fitness_file}")

    # Gather main trees:
    sim_name = config['name']
    sim_params = config['parameters']
    sim_dims = config['dims']
    sim_subs = config['substrates']
    sim_fit = config['fit']

    # Build directives:
    directives_min = list()
    directives_max = list()
    for dim in sim_dims:
        # Build the directive:
        directive_min = [dim['min']] * dim['size']
        directive_max = [dim['max']] * dim['size']
        # Append the directive:
        directives_min.extend(directive_min)
        directives_max.extend(directive_max)
    directives = [directives_min, directives_max]
    logging.info(f"[i] Directives built: {directives}")
    directives = tf.convert_to_tensor(directives, dtype=tf.float32)

    # Fitness test:
    logging.info("[i] Testing fitness...")
    try:
        fitness_function(directives)
        logging.info(f"[i] Fitness tested in limits. Success.")
    except Exception as ex:
        logging.error(f"[x] Exception found while running fitness: {ex}\n\nFix your fitness function before running it.")

    # Build the directives vector:
    substrates = list()
    for sub in sim_subs:
        # Get the substrate:
        substrate = sub['substrate_name']
        # Drop key 'substrate_name':
        sub.pop('substrate_name')
        # Add the substrate to the simulation:
        try:
            substrate_instance = getattr(tensorcro, substrate)
            logging.info(f"[i] Substrate built successfully: {substrate}")
        except AttributeError:
            logging.error(f"[x] Substrate not found: {substrate}")
            raise AttributeError(f"Substrate not found: {substrate}")
        substrates.append(substrate_instance)

    # Build TensorCro:
    t_cro = tensorcro.TensorCro(**sim_params, subs=substrates)
    logging.info("[i] TensorCro built successfully.")

    # Build Callback:
    save_callback = SimulationSaveCallback(base_path)

    # Run fitness:
    logging.info(f"[i] Optimizing {sim_name}...")
    try:
        _res_ = t_cro.fit(fitness_function, directives, max_iter=sim_fit["iterations"], device='/GPU:0',
                          seed=sim_fit["seed"], shards=sim_fit["shards"], save=False, time_limit=sim_fit["time_limit"],
                          tf_compile=sim_fit["compile"], callback=save_callback, minimize=sim_fit["minimize"])
    except Exception as ex:
        logging.error(f"[x] Error while optimizing: {ex}")
        if save_callback is not None:
            save_callback.exception_handler(ex)
        raise RuntimeError(f"[x] Error while optimizing: {ex}")
    solution, fitness_values = _res_

    # Print results:
    logging.info(f"[i] Optimization finished:\n\tSolutions:\n{solution.numpy()}\n\tFitness:\n{fitness_values.numpy()}")
    return


class SimulationSaveCallback:
    def __init__(self, base_path: str):
        """
        """
        self.base_path = base_path
        self.fitness_history = list()
        self.current_shard = 0
        self.best_fitness = -np.inf
        self.last_best_fitness: None | np.ndarray = None
        self.last_best_pop: None | np.ndarray = None

    def exception_handler(self, exception):
        """
        :param exception: The exception raised.
        :return:
        """
        if self.last_best_pop is not None:
            np.save(os.path.join(self.base_path, 'population_checkpoint.npy'), self.last_best_pop)
        logging.error(f'[x] Error in Callback: {exception}')

    def __call__(self, *args, **kwargs):
        """
        :param args: Arguments of the callback. [0] is the population, [1] is the fitness [2] is the max shards.
        :param kwargs: Not used.
        :return: True if the callback has been called correctly.
        """
        sorted_fitness = args[1]
        max_shards = args[2]
        best_fitness = float(sorted_fitness[0])
        timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        self.fitness_history.append(best_fitness)
        self.max_shards = max_shards
        # Update shard and best fitness.
        self.current_shard += 1
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.last_best_fitness = self.current_shard
            self.last_best_pop = args[0].numpy()

        if self.last_best_pop is not None:
            np.save(os.path.join(self.base_path, f'population_{timestamp}.npy'), self.last_best_pop)
        return True













# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    docker_run()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
