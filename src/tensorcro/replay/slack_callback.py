# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import slack
import time
import os
import numpy as np
import tensorflow as tf


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SlackCallback:
    def __init__(self, bot_token: str, channel: str, username: str,
                 icon_emoji: str = ':robot_face:', icon_url: str = None):
        """
        Slack callback class. This class is used to send messages to a slack channel. It is used by the fit method
        to send messages to a slack channel when a shard is finished.
        :param bot_token: Slack bot token.
        :param channel: Slack channel to send the messages.
        :param username: Username to send the messages.
        :param icon_emoji: Emoji to use as icon.
        :param icon_url: URL to use as icon.
        """
        self.client = slack.WebClient(token=bot_token)
        self.token = bot_token
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        self.icon_url = icon_url
        self.current_shard = 0
        self.best_fitness = -np.inf
        self.last_best_fitness = 0
        self.last_message = None
        self.initialization_ts = time.time()

        # Display fitness:
        self.fitness_history = list()

    def exception_handler(self, exception):
        """
        This method is called when an exception is raised. It sends a message to the slack channel to tell that
        an exception has been raised.
        :param exception: The exception raised.
        :return:
        """
        elapsed_time = time.time() - self.initialization_ts
        elapsed_time_hrs = int(elapsed_time // 3600)
        elapsed_time_mins = int((elapsed_time % 3600) // 60)
        elapsed_time_secs = elapsed_time % 60
        elapsed_time_days = int(elapsed_time_hrs // 24)
        __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        __all_text__ = (f'[{__timestamp}] *@{self.username}* \n{self.icon_emoji}: Oops, an exception raised: '
                        f'{exception}'
                        f'\n\t:robot_face: The simulation took {self.current_shard} shards and'
                        f' {elapsed_time_days} days, {elapsed_time_hrs} hours, {elapsed_time_mins} minutes and'
                        f' {elapsed_time_secs: .02f} seconds.\n')
        self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)

    def end(self, best_solution: (np.ndarray, tf.Tensor) = None):
        """
        This method is called when the fit method is finished. It sends a message to the slack channel to tell that
        the fit method is finished.
        :return:
        """
        elapsed_time = time.time() - self.initialization_ts
        elapsed_time_hrs = int(elapsed_time // 3600)
        elapsed_time_mins = int((elapsed_time % 3600) // 60)
        elapsed_time_secs = elapsed_time % 60
        elapsed_time_days = int(elapsed_time_hrs // 24)
        __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        __all_text__ = (f':raised_hands: :trophy: :sports_medal: :sparkler: :sports_medal: :trophy: :raised_hands:\n'
                        f'[{__timestamp}] *@{self.username}* \n{self.icon_emoji}: The simulation has finished!\n'
                        f'\t:trophy: Best fitness: {self.best_fitness}\n'
                        f'\t:coral: Last best found in shard {self.last_best_fitness}.\n'
                        f'\t:robot_face: The simulation took {self.current_shard} shards and'
                        f' {elapsed_time_days} days, {elapsed_time_hrs} hours, {elapsed_time_mins} minutes and'
                        f' {elapsed_time_secs: .02f} seconds.\n'
                        f':raised_hands: :trophy: :sports_medal: :sparkler: :sports_medal: :trophy: :raised_hands:')
        # self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)
        if best_solution is not None:
            if isinstance(best_solution, tf.Tensor):
                best_solution = best_solution.numpy()
            np.save('best_solution.npy', best_solution)
            self.client.files_upload(channels=self.last_message.data['channel'],
                                     file='best_solution.npy',
                                     filename='last_population.npy',
                                     initial_comment=__all_text__)
            os.remove('best_solution.npy')

    def __call__(self, *args, **kwargs):
        """
        This method is called when the callback is called. It sends a message to the slack channel.
        :param args: Arguments of the callback. [0] is the population, [1] is the fitness [2] is the max shards.
        :param kwargs:
        :return:
        """
        sorted_population = args[0]
        sorted_fitness = args[1]
        max_shards = args[2]

        best_fitness = float(sorted_fitness[0])
        best_individual = sorted_population[0].numpy()

        self.fitness_history.append(best_fitness)

        # Update shard and best fitness.
        self.current_shard += 1
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.last_best_fitness = self.current_shard
            __generated_text = (f' Shard {self.current_shard} finished! ({100 * self.current_shard / max_shards: .01f}'
                                f'%) \n'
                                f'\t:bicyclist: Best fitness: {best_fitness}\n'
                                f'\t:coral: Best individual: {best_individual}\n'
                                f'\t:trophy: WE FOUND A NEW BEST!')
        else:
            __generated_text = (f' Shard {self.current_shard} finished!\n'
                                f'\t:bicyclist: Best fitness: {best_fitness}\n'
                                f'\t:coral: Best individual: {best_individual}\n'
                                f'\t:microscope: Last best found in shard {self.last_best_fitness}.')

        # Generate timestamp with day, month, year, hour, minute and second.
        __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        __all_text__ = f'[{__timestamp}] *{self.username}* \n{self.icon_emoji}:{__generated_text}'
        # Send message to slack.
        if self.last_message is not None:
            self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                                    ts=self.last_message.data['ts'])

        self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)
        return True
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
