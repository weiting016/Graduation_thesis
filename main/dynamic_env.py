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
        self.action_penalty_d = None #record added action with its penalty in diction form
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
        self.seed(2)
        self.observation_space = frequencies
        self.action_space = Discrete(len(self.motions))
        self.choice_positions =['va','pp','po','sib']
        self.positions = np.random.choice(self.choice_positions, size=1)
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
            action_penalty = self.get_action_reward(motion,self.action_penalty_d)

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
        self.positions = np.random.choice(self.choice_positions)#(['va','pp','po','sib'])
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
    
    def get_action_reward(self, action,add_penalty_dict = None):
        penalty = 0
        if action == 'met kracht tegen- of vastgehouden' or action == 'afzondering (deur op slot)':
            penalty = -2
        elif action == 'client afgeleid' or action == 'contact beeindigd/weggegaan' or action == 'naar andere kamer/ruimte gestuurd':
            penalty = -1
        if add_penalty_dict != None:
            penalty.get(action)

        return penalty

    
    def set_flag(self):
        self.flag = True
    
        return 
    
    
    def drift(self,add_actions=0,add_states=0):
        """
        change of the env: the observation space: add actions and states
        assign new transition probability
        """
        if self.flag == True:
#state action总数不变，只是部分path关闭
            print("drift happen")
            #self.observation_space  = change_frequencies(self.motions, self.states)
    
            if add_actions != 0:
                syn_actions = syn_state_action('a',add_actions)#['a1','a2','a3']
                print(syn_actions)

                self.motions.extend(syn_actions)
                self.action_penalty_d = action_penalty_dict(syn_actions)
            if add_states != 0:
                syn_states = syn_state_action('s',add_states)
                self.states.extend(syn_states)
                self.choice_positions.extend(syn_states)
                change_states_severity(self,syn_states)

            self.observation_space  = change_frequencies(self.motions, self.states)
            self.action_space = Discrete(len(self.motions))

            """self.severity = {'va': 0.0, 
                            'po': -1.0,
                            'sib': -3.0,
                            'pp' : -4.0,
                            'Tau': 1.0,
                            's1': 0.0,
                            's2': 1.0,
                            's3': 1}  #gained reward when reach certain state"""   
        return



def change_frequencies(actions,states):
   # print('execute change frequencies')
    new_freq = pd.DataFrame(index = actions)
    action_num = len(actions)
    for s in states:
        #new_freq[s] = list(map(create_transition, repeat(states,action_num)))
        new_freq[s] = list(map(drift_transition, repeat(states,action_num)))
   # print(new_freq.shape)
    return new_freq
    

def create_transition(states): 
    """
    create new transition matrix with the added states and actions
    input spacify- different type of drift 
    like Q-learning alpha
    type of drift
    """
    #print("excute create transitions ")
    np.random.seed(2) #set the random seed make result reproducable
    possible_path = len(states) #possible transitions number
    path_num = random.randint(1, possible_path+1)  #assigned real path
    trans_prob = np.random.dirichlet(np.ones(path_num), size=1)[0] #assign transition probability 可能长度不一样

    if path_num < possible_path:
        np.concatenate((trans_prob),[0]*(possible_path- path_num))
        trans_prob = list(trans_prob)
        trans_prob.extend([0]*(possible_path- path_num))
        
    np.random.shuffle(trans_prob) #possible closed pass can appear random posation
    trans_dict = {s: t for s, t in zip(states, trans_prob)}
   # print(trans_dict)
    return trans_dict



def drift_transition(states):
    """
    the drift happen in transition probability
    if close = True prob can become 0 (path close),otherwise just change transition probability
    do not influence the size state and action space
    """
    #print('excute drift transition')
    np.random.seed(2) #set the random seed make result reproducabl
    possible_path = len(states) #possible transitions number
    #if self.close == True:
    path_num = random.randint(1, possible_path)  #assigned real path,less than orignal pathes
    trans_prob = np.random.dirichlet(np.ones(path_num), size=1)[0] #assign transition probability 可能长度不一样
   # print(type(trans_prob))
   # print("trans_prob:", trans_prob)
    if path_num < possible_path:
        np.concatenate(((trans_prob),[0]*(possible_path- path_num)))
        trans_prob = list(trans_prob)
        trans_prob.extend([0]*(possible_path- path_num))
        
    np.random.shuffle(trans_prob) #possible closed pass can appear random posation
    trans_dict = {s: t for s, t in zip(states, trans_prob)}
   # print(trans_dict)
    return trans_dict

def action_penalty_dict(add_action):
    numadd = len(add_action)
    penalty = np.random.randint(-5.0,0,numadd)
    return {s: t for s, t in zip(add_action, penalty)}
    
def syn_state_action(prefix,n):
    return [f"{prefix}{i}" for i in range(n)]

def change_states_severity(self,add_states):
    add_num = len(add_states)
    add_severity = np.random.randint(-5.0,5.0,add_num) #define severity with in range -5.0 - 5.0
    severity_dict = {s: t for s, t in zip(add_states, add_severity)}
    self.severity.update(**severity_dict)
    return