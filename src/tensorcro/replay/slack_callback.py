# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import slack
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class SlackCallback:
    def __init__(self, bot_token: str, channel: str, username: str, simulation_name: str = None):
        """
        Slack callback class. This class is used to send messages to a slack channel. It is used by the fit method
        to send messages to a slack channel when a shard is finished.
        :param bot_token: Slack bot token.
        :param channel: Slack channel to send the messages.
        :param username: Username to send the messages.
        """
        # Create client:
        self.client = slack.WebClient(token=bot_token)
        # Fetch users:
        try:
            users = self.client.users_list()['members']
            for user in users:
                if user['profile']['display_name'] == username:
                    username = user['id']
                    break
        except Exception as e:
            logging.error(f'[SlackBot] Error fetching users: {e}')
        # Initialize variables:
        if simulation_name is not None:
            self.simulation_name = simulation_name
        else:
            self.simulation_name = 'TensorCRO'
        self.token = bot_token
        self.channel = channel
        self.username = username
        self.current_shard = 0
        self.best_fitness = -np.inf
        self.last_best_fitness = 0
        self.last_message = None
        self.initialization_ts = time.time()
        # Display fitness:
        self.fitness_history = list()
        self.ts_del = None
        self.max_shards = None
        self.last_best_pop = None

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
        timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        # Create block.
        block = {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"\n:warning::no_entry::warning::boom::warning::boom::warning::no_entry::warning:\n"
                                f":coral:: *<@{self.username}>*, there was an error in your "
                                f"`{self.simulation_name}` simulation."
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f":cake: *Shards:*"
                                    f"\n`{self.current_shard}/{self.max_shards} "
                                    f"({100 * self.current_shard / self.max_shards:.1f}%)` :warning:"
                        },
                        {
                            "type": "mrkdwn",
                            "text": ":clock9:*Timestamp:*\n"
                                    f"`{timestamp}`"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":trophy: *Best fitness:*\n`{self.best_fitness :.3f}`"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":telescope: *Error found:*\n`In shard {self.current_shard + 1}`"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": ":alarm_clock: *Elapsed time*:"
                    },
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Days:* {elapsed_time_days}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Minutes:* {elapsed_time_mins}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Hours:* {elapsed_time_hrs}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Seconds:* {elapsed_time_secs:.2f}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f'*Exception raised*:\n```{exception}```\n'
                                f':warning::no_entry::warning::boom::warning::boom::warning::no_entry::warning:'
                    }
                }
            ]
        }
        try:
            # Send message to slack.
            np.save('./best_tmp.npy', self.last_best_pop)
            if self.last_message is not None:
                self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                                        ts=self.last_message.data['ts'])
                if self.ts_del is not None:
                    self.client.files_delete(token=self.token, channel=self.last_message.data['channel'],
                                             ts=self.ts_del['ts'], file=self.ts_del['file']['id'])
                    # self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                    #                         ts=self.ts_del['ts'])
            self.last_message = self.client.chat_postMessage(channel=self.channel, blocks=block['blocks'])
            self.ts_del = self.client.files_upload(channels=self.last_message.data['channel'],
                                                   file='./best_tmp.npy',
                                                   filename='Backup.npy')
            os.remove('./best_tmp.npy')
            return True
        except Exception as e:
            logging.error(f'[SlackBot] Error sending exception to slack: {e}')
            return False

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
        timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
        # Create fitness plot.
        plt.plot(self.fitness_history, color='blue', linewidth=1.5, linestyle='solid', marker='o')
        plt.title(f'Fitness function ({self.current_shard} shards)')
        plt.xlabel('Shards')
        plt.xticks(np.arange(0, self.current_shard, 1))
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.savefig('./fitness_tmp.png')
        # Create block.
        block = {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":raised_hands: :trophy: :sports_medal: :sparkler: :sports_medal: :trophy: \n"
                                f":coral:: <@{self.username}>, your `{self.simulation_name}` simulation has finished."
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f":cake: *Shards:*"
                                    f"\n{self.current_shard}/{self.max_shards} "
                                    f"({100 * self.current_shard / self.max_shards:.1f}%)"
                        },
                        {
                            "type": "mrkdwn",
                            "text": ":clock9: *Timestamp:*\n"
                                    f"{timestamp}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":trophy: *Best fitness:*\n{self.best_fitness :.3f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":telescope: *Last improvement:*\nIn shard {self.last_best_fitness}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": ":alarm_clock: *Elapsed time:*"
                    },
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Days:* {elapsed_time_days}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Minutes:* {elapsed_time_mins}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Hours:* {elapsed_time_hrs}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Seconds:* {elapsed_time_secs:.2f}"
                        }
                    ]
                }
            ]
        }
        try:
            # Send message to slack.
            np.save('./solution.npy', best_solution)
            if self.last_message is not None:
                self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                                        ts=self.last_message.data['ts'])
                if self.ts_del is not None:
                    self.client.files_delete(token=self.token, channel=self.last_message.data['channel'],
                                             ts=self.ts_del['ts'], file=self.ts_del['file']['id'])
                    # self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                    #                         ts=self.ts_del['ts'])
            self.last_message = self.client.chat_postMessage(channel=self.channel, blocks=block['blocks'])
            if self.current_shard > 1:
                self.ts_del = self.client.files_upload(channels=self.last_message.data['channel'],
                                                       file='./fitness_tmp.png',
                                                       filename='Fitness.png')
            self.ts_del = self.client.files_upload(channels=self.last_message.data['channel'],
                                                   file='./solution.npy',
                                                   filename='Solution.npy')
            os.remove('./fitness_tmp.png')
            os.remove('./solution.npy')
            return True
        except Exception as e:
            logging.error(f'[SlackBot] Error sending exception to slack: {e}')
            return False

    def __call__(self, *args, **kwargs):
        """
        This method is called when the callback is called. It sends a message to the slack channel.
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
        # Create fitness plot.
        fig = plt.figure()
        fig.plot(self.fitness_history, color='blue', linewidth=1.5, linestyle='solid', marker='o')
        fig.title(f'Fitness function ({self.current_shard} shards)')
        fig.xlabel('Shards')
        fig.xticks(np.arange(0, self.current_shard, 1))
        fig.ylabel('Fitness')
        fig.grid(True)
        fig.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        fig.savefig('./fitness_tmp.png')
        # Create block.
        block = {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":coral:: *<@{self.username}>*, there is news about your "
                                f"`{self.simulation_name}` simulation."
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f":cake: *Shards:*"
                                    f"\n{self.current_shard}/{max_shards} "
                                    f"({100 * self.current_shard / max_shards:.01f}%)"
                        },
                        {
                            "type": "mrkdwn",
                            "text": ":clock9: *Timestamp:*\n"
                                    f"{timestamp}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":trophy: *Best fitness:*\n{self.best_fitness :.03f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f":telescope: *Last improvement:*\nIn shard {self.last_best_fitness}"
                        }
                    ]
                }
            ]
        }
        try:
            # Send message to slack.
            if self.last_message is not None:
                self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                                        ts=self.last_message.data['ts'])
                if self.ts_del is not None:
                    self.client.files_delete(token=self.token, channel=self.last_message.data['channel'],
                                             ts=self.ts_del['ts'], file=self.ts_del['file']['id'])
                    # self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
                    #                         ts=self.ts_del['ts'])
            self.last_message = self.client.chat_postMessage(channel=self.channel, blocks=block['blocks'])
            if self.current_shard > 1:
                self.ts_del = self.client.files_upload(channels=self.last_message.data['channel'],
                                                       file='./fitness_tmp.png',
                                                       filename='Fitness.png')
            os.remove('./fitness_tmp.png')
            return True
        except Exception as e:
            logging.error(f'[SlackBot] Error sending message to slack: {e}')
            self.last_message = None
            return False


# def __call___v0(self, *args, **kwargs):
#     """
#     This method is called when the callback is called. It sends a message to the slack channel.
#     :param args: Arguments of the callback. [0] is the population, [1] is the fitness [2] is the max shards.
#     :param kwargs:
#     :return:
#     """
#     sorted_population = args[0]
#     sorted_fitness = args[1]
#     max_shards = args[2]
#
#     best_fitness = float(sorted_fitness[0])
#     best_individual = sorted_population[0].numpy()
#
#     self.fitness_history.append(best_fitness)
#
#     # Update shard and best fitness.
#     self.current_shard += 1
#     if best_fitness > self.best_fitness:
#         self.best_fitness = best_fitness
#         self.last_best_fitness = self.current_shard
#         __generated_text = (f' Shard {self.current_shard} finished! ({100 * self.current_shard / max_shards: .01f}'
#                             f'%) \n'
#                             f'\t:bicyclist: Best fitness: {best_fitness}\n'
#                             f'\t:coral: Best individual: {best_individual}\n'
#                             f'\t:trophy: WE FOUND A NEW BEST!')
#     else:
#         __generated_text = (f' Shard {self.current_shard} finished!\n'
#                             f'\t:bicyclist: Best fitness: {best_fitness}\n'
#                             f'\t:coral: Best individual: {best_individual}\n'
#                             f'\t:microscope: Last best found in shard {self.last_best_fitness}.')
#
#     # Generate timestamp with day, month, year, hour, minute and second.
#     __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
#     __all_text__ = f'[{__timestamp}] *{self.username}* \n{self.icon_emoji}:{__generated_text}'
#     # Send message to slack.
#     if self.last_message is not None:
#         self.client.chat_delete(token=self.token, channel=self.last_message.data['channel'],
#                                 ts=self.last_message.data['ts'])
#
#     self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)
#     return True

# def exception_handler_v0(self, exception):
#     """
#     This method is called when an exception is raised. It sends a message to the slack channel to tell that
#     an exception has been raised.
#     :param exception: The exception raised.
#     :return:
#     """
#     elapsed_time = time.time() - self.initialization_ts
#     elapsed_time_hrs = int(elapsed_time // 3600)
#     elapsed_time_mins = int((elapsed_time % 3600) // 60)
#     elapsed_time_secs = elapsed_time % 60
#     elapsed_time_days = int(elapsed_time_hrs // 24)
#     __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
#     __all_text__ = (f'[{__timestamp}] *@{self.username}* \n{self.icon_emoji}: Oops, an exception raised: '
#                     f'{exception}'
#                     f'\n\t:robot_face: The simulation took {self.current_shard} shards and'
#                     f' {elapsed_time_days} days, {elapsed_time_hrs} hours, {elapsed_time_mins} minutes and'
#                     f' {elapsed_time_secs: .02f} seconds.\n')
#     self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)

# def end_v0(self, best_solution: (np.ndarray, tf.Tensor) = None):
#     """
#     This method is called when the fit method is finished. It sends a message to the slack channel to tell that
#     the fit method is finished.
#     :return:
#     """
#     elapsed_time = time.time() - self.initialization_ts
#     elapsed_time_hrs = int(elapsed_time // 3600)
#     elapsed_time_mins = int((elapsed_time % 3600) // 60)
#     elapsed_time_secs = elapsed_time % 60
#     elapsed_time_days = int(elapsed_time_hrs // 24)
#     __timestamp = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime())
#     __all_text__ = (f':raised_hands: :trophy: :sports_medal: :sparkler: :sports_medal: :trophy: :raised_hands:\n'
#                     f'[{__timestamp}] *@{self.username}* \n{self.icon_emoji}: The simulation has finished!\n'
#                     f'\t:trophy: Best fitness: {self.best_fitness}\n'
#                     f'\t:coral: Last best found in shard {self.last_best_fitness}.\n'
#                     f'\t:robot_face: The simulation took {self.current_shard} shards and'
#                     f' {elapsed_time_days} days, {elapsed_time_hrs} hours, {elapsed_time_mins} minutes and'
#                     f' {elapsed_time_secs: .02f} seconds.\n'
#                     f':raised_hands: :trophy: :sports_medal: :sparkler: :sports_medal: :trophy: :raised_hands:')
#     # self.last_message = self.client.chat_postMessage(channel=self.channel, text=__all_text__)
#     if best_solution is not None:
#         if isinstance(best_solution, tf.Tensor):
#             best_solution = best_solution.numpy()
#         np.save('best_solution.npy', best_solution)
#         self.client.files_upload(channels=self.last_message.data['channel'],
#                                  file='best_solution.npy',
#                                  filename='last_population.npy',
#                                  initial_comment=__all_text__)
#         os.remove('best_solution.npy')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
