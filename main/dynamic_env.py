from typing import List, Tuple

import ast
import numpy as np
import pandas as pd
import gymnasium 

from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import random
from itertools import repeat


class TaskEnv_drift():
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
        self.flag = False #drift indicator
        self.severity = {'va': 0.0,
                         'po': -1.0,
                         'sib': -3.0,
                         'pp' : -4.0,
                         'Tau': 1.0} #different reward when reach different state
        
        self.motions = ['contact beeindigd/weggegaan',
          'client toegesproken/gesprek met client',
          'geen',
          'client afgeleid',
          'naar andere kamer/ruimte gestuurd',
          'met kracht tegen- of vastgehouden',
          'afzondering (deur op slot)'] #action
        self.states = ['va','sib','pp','po','Tau'] 
        
        frequencies = pd.read_csv("frequencies.csv", index_col=0)
        for label in frequencies:
            for action in self.motions:
                frequencies[label][action] = ast.literal_eval(frequencies[label][action]) #判断需要计算的内容是不是合法的Python类型，如果是则执行，否则就报错
        #print("original frequencies",frequencies.shape)
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
       # print("execute step function")
        motion = self.motions[action]
        current_position = self.positions
        new_position = self.get_next_state(self.observation_space[current_position][motion])[0]
        transition_reward = self.severity[new_position]
        if self.flag == False: #without drift
            action_penalty = self.get_action_reward(motion)
        else: #add drift, use extended action_reward function
            action_penalty = self.get_action_reward_drift(motion)

        self.positions = new_position
        self.episode_actions.append((action, current_position, new_position))

        #print("transition_reward",transition_reward)
       #print("action_penalty",action_penalty)

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
       # print("current state_action", state_action.items())
        for key, value in state_action.items():
            states.append(key)
            prob.append(value)

        p = np.asarray(prob).astype('float64')
         #normalize prob incase it doesnt sum to 1
        p /= p.sum()
       # print("transation prob",p)

        return np.random.choice(states, size=1, p=p)
    
    def get_action_reward(self, action):
        penalty = 0
        if action == 'met kracht tegen- of vastgehouden' or action == 'afzondering (deur op slot)':
            penalty = -2
        elif action == 'client afgeleid' or action == 'contact beeindigd/weggegaan' or action == 'naar andere kamer/ruimte gestuurd':
            penalty = -1
        return penalty

    
    def set_flag(self):
        self.flag = True
        return 
    
    def drift(self):
        """
        change of the env: the observation space: add actions and states
        assign new transition probability
        """
        if self.flag == True:

            syn_actions = ['a1','a2','a3']
            syn_states = ['s1','s2','s3']
            self.motions.extend(syn_actions)
            self.states.extend(syn_states)
            self.observation_space  = added_frequencies(self.motions, self.states)
            self.action_space = Discrete(len(self.motions))
            self.severity = {'va': 0.0, 
                            'po': -1.0,
                            'sib': -3.0,
                            'pp' : -4.0,
                            'Tau': 1.0,
                            's1': 0.0,
                            's2': 1.0,
                            's3': 1}  #gained reward when reach certain state
        
        return
    
    def reset_drift(self) -> str:
        """Resets the environment. The agent will be transferred to a random location on the map. The goal stays the same and the timer is set to 0.

        Returns:
            Tuple[int, int]: The initial position of the agent.
        """
        self.timer = 0
        self.positions = np.random.choice(['va','pp','po','sib','s1','s2','s3'])
        self.episode_actions = []
        return self.positions

    def get_action_reward_drift(self, action):
        penalty = 0
        if action == 'met kracht tegen- of vastgehouden' or action == 'afzondering (deur op slot)':
            penalty = -2
        elif action == 'client afgeleid' or action == 'contact beeindigd/weggegaan' or action == 'naar andere kamer/ruimte gestuurd':
            penalty = -1

        #add new state with different reward 
        elif action =='a1' or action == 'a2':
            penalty = 1
        elif action == 'a3':
            penalty = -3
        return penalty

def added_frequencies(actions,states):
    new_freq = pd.DataFrame(index = actions)
    action_num = len(actions)
    for s in states:
        new_freq[s] = list(map(create_transition, repeat(states,action_num)))
    #print(new_freq.shape)
    return new_freq
    
#def create_transition_closepath(states):
    #"""disable certain path to certain state"""

def create_transition(states): 
    """
    create new transition matrix with the added states and actions
    input spacify- different type of drift 
    like Q-learning alpha
    type of drift
    """

    np.random.seed(2) #set the random seed make result reproducable
    possible_path = len(states) #possible transitions number
    path_num = random.randint(1, possible_path+1)  #assigned real path
    trans_prob = np.random.dirichlet(np.ones(path_num), size=1)[0] #assign transition probability 可能长度不一样

    if path_num < possible_path:
        #np.concatenate((trans_prob),[0]*(possible_path- path_num))
        trans_prob = list(trans_prob)
        trans_prob.extend([0]*(possible_path- path_num))
        
    np.random.shuffle(trans_prob) #possible closed pass can appear random posation
    #keys = states
    #valuess = trans_prob
    trans_dict = {s: t for s, t in zip(states, trans_prob)}
   # print(trans_dict)
    return trans_dict


