from typing import List, Tuple

import ast
import numpy as np
import pandas as pd
import gymnasium as gym
from gym.spaces import Discrete
from gym.utils import seeding
import random


class TaskEnv():
    def __init__(self,
                 time_out: int = 6,
                 timeout_reward=-1,
                 goal_reward=1,
                 time_reward_multiplicator=1):
        """Contructor for the TaskEnvironment

        Args:
            size (int, optional): The size of the maze. Defaults to 15.
            time_out (int, optional): Time to explore the maze before the game is over. Defaults to 100.
        """
        self.severity = {'va': 0.0,
                         'po': -1.0,
                         'sib': -3.0,
                         'pp' : -4.0,
                         'Tau': 1.0}
        self.motions = ['contact beeindigd/weggegaan',
          'client toegesproken/gesprek met client',
          'geen',
          'client afgeleid',
          'naar andere kamer/ruimte gestuurd',
          'met kracht tegen- of vastgehouden',
          'afzondering (deur op slot)']
        frequencies = pd.read_csv("frequencies_add.csv", index_col=0)
        for label in frequencies:
            for action in self.motions:
                frequencies[label][action] = ast.literal_eval(frequencies[label][action])
        self.goal = ['Tau']
        self.time_out = time_out
        self.timer = 0
        self.timeout_reward = timeout_reward
        self.goal_reward = goal_reward
        self.time_reward_multiplicator = time_reward_multiplicator
        self.seed()
        self.observation_space = frequencies
        self.action_space = Discrete(len(self.motions))
        self.positions = np.random.choice(['va','pp','po','sib'], size=1)
        self.episode_actions = []

    def step(self, action: int, info = False):
        motion = self.motions[action]
        current_position = self.positions
        new_position = self.get_next_state(self.observation_space[current_position][motion])[0]
        transition_reward = self.severity[new_position]
        action_penalty = self.get_action_reward(motion)
        self.positions = new_position
        self.episode_actions.append((action, current_position, new_position))
        if self._is_timeout():
            reward = self.timeout_reward 
            done = True
        elif new_position == 'Tau':
            reward = transition_reward + action_penalty
            done = True
        else:
            reward = transition_reward + action_penalty
            done = False
        self.timer += 1
        if done and info:
            return self.positions, reward, done, self.episode_actions
        else:
            return self.positions, reward, done, []

    def reset(self) -> str:
        """Resets the environment. The agent will be transferred to a random location on the map. The goal stays the same and the timer is set to 0.

        Returns:
            Tuple[int, int]: The initial position of the agent.
        """
        self.timer = 0
        self.positions = np.random.choice(['va','pp','po','sib'])
        self.episode_actions = []
        return self.positions
                                          
    def _is_timeout(self) -> bool:
        """Checks whether the environment has reached its timeout.

        Returns:
            bool: True for timeout is exceeded and false if not.
        """
        return self.timer >= self.time_out

    def seed(self, seed: int = None) -> List[int]:
        """Ensures reproductability

        Args:
            seed (int, optional): A seed number. Defaults to None.

        Returns:
            List[int]: The seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
       
    def get_next_state(self, state_action):
        states = []
        prob = []
        for key, value in state_action.items():
            states.append(key)
            prob.append(value)
        return np.random.choice(states, size=1, p=prob)
    
    def get_action_reward(self, action):
        penalty = 0
        if action == 'met kracht tegen- of vastgehouden' or action == 'afzondering (deur op slot)':
            penalty = -2
        elif action == 'client afgeleid' or action == 'contact beeindigd/weggegaan' or action == 'naar andere kamer/ruimte gestuurd':
            penalty = -1
        return penalty

