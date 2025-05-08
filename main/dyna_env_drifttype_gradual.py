from typing import List, Tuple
import ast
import numpy as np
import pandas as pd
import gymnasium 
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import random
from itertools import repeat

"""gradual drift - 在step时模型随机在两个observation space中选择"""

class TaskEnv_driftype_gradual(gymnasium.Env):
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
        self.observation_space_drift = None #store drifted distribution  for gradual drift
        self.action_space = Discrete(len(self.motions))
        self.choice_positions =['va','pp','po','sib']
        self.positions = np.random.choice(self.choice_positions, size=1)
        self.episode_actions = []
        self.drift_type = None
        self.drift_swap = False

    def step(self, action: int, info = False):
       # print("execute step function")
        motion = self.motions[action]
        current_position = self.positions
        #introduce function here, t=drift, which MDP used
        observation_space = self.observation_space
        if self.drift_type == 'gradual':
            if self.drift_swap==False:
            #print('drift gradual happen',type(self.observation_space),type(self.observation_space_drift))
                observation_space = random.choice([self.observation_space,self.observation_space_drift])
            #print(type(observation_space))
            else:
                 observation_space = self.observation_space_drift
    
        new_position = self.get_next_state(observation_space[current_position][motion])[0]
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
            add_penalty_dict.get(action)

        return penalty

    
    def set_flag(self):
        self.flag = True
    
        return 

    """drift control
    add_actions:number of new actions
    change_at_states: which of the states the transition probability change
    drift_dis_type: random/simple/reverse,transition prob change randomly/similar with original distribution / reversed distribution
"""

    def drift(self, add_actions=0, change_at_states=None, drift_dis_type='random', intensity=0.5, drift_type='sudden'):
        """
        change of the env: the observation space: add actions and states
        assign new transition probability
        """
        if self.flag == True:
            # state action总数不变，只是部分path关闭
            self.drift_type = drift_type
            print("drift happen")
            if add_actions != 0:
                syn_actions = syn_state_action('a', add_actions)  # ['a1','a2','a3']
                #print(syn_actions)
                self.motions.extend(syn_actions)
                self.action_penalty_d = action_penalty_dict(syn_actions)
            
    
            global change_frequencies
            if drift_type == 'sudden':
                self.observation_space = change_frequencies(
                    self.motions, 
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )
            elif drift_type == 'gradual':
                 self.observation_space_drift = change_frequencies(
                    self.motions, 
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )
        
            
        """
            self.observation_space = change_frequencies(
                self.motions,
                change_at_states,
                original_freq=self.observation_space, 
                drift_dis_type=drift_dis_type, 
                intensity=intensity
            )
            self.action_space = Discrete(len(self.motions))
            """
        return


def drift_transition(states):
        """
        the drift happen in transition probability
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

def perturb_probs(orig_probs,intensity=0.5, preserve_rank=True, random_seed=2):
    """
    生成一个新的概率分布，基于原始概率分布进行扰动。

    参数:
    intensity 可调整参数，控制扰动强度（0:完全原始分布，1:完全扰动分布）
    orig_probs (list or np.array): 原始概率分布，必须是一个合法的概率分布（和为1，非负）。
    preserve_rank (bool): 是否保持原始概率的大小顺序。如果为False，则反转大小顺序。
    random_seed (int or None): 随机种子，用于确保结果可复现。如果为None，则不设置种子。

    返回:
    np.array: 扰动后的新概率分布。
    """
    orig_probs = np.array(orig_probs, dtype=float)
    n = len(orig_probs)
    print(np.sum(orig_probs),orig_probs)
    # 验证输入是否为合法的概率分布,不验证保证输出的是sum = 1即可
    #if not np.allclose(np.sum(orig_probs), 1.0) or np.any(orig_probs < 0):
        #raise ValueError("invalid value")
    
    perturbations = np.random.uniform(0, 1, n)
    perturbations = np.sort(perturbations)[::-1] if np.all(np.diff(orig_probs) <= 0) else np.sort(perturbations)
    # normalize, make sure sum=1
    perturbations /= np.sum(perturbations)
    #print(type(perturbations),type(intensity),intensity)

    new_probs = (1 - intensity) * orig_probs + intensity * perturbations
    new_probs /= np.sum(new_probs)
    if preserve_rank == False:
        #print('reversed')
        new_probs = new_probs[::-1]
        #print(new_probs)
    print('new prob',np.sum(new_probs))
    return new_probs.tolist()

def change_frequencies(actions, change_at_states, original_freq=None, 
                        drift_dis_type='random', intensity=0.5):
        """
        Generates a new transition probability matrix with controlled drift behavior.
        Only modifies probabilities for specified states, leaving others unchanged.
        """
        np.random.seed(2)  # Maintain reproducibility  
        if original_freq is None:
            raise ValueError("Original frequencies must be provided")       
        new_freq = original_freq.copy(deep=True)
        
        for s in change_at_states:
            if drift_dis_type == "random":
                # Original behavior: fully random
                new_freq[s] = list(map(drift_transition, repeat(change_at_states, len(actions))))
            else:
                # Controlled drift based on original probabilities
                new_probs = []
                for a in actions:
                    orig_probs = list(original_freq[s][a].values())
                    if drift_dis_type == "similar":
                        # Perturb probabilities while preserving rank
                        new_p = perturb_probs(orig_probs, intensity, preserve_rank=True)
                    elif drift_dis_type == "inverse":
                        # Invert probability ranks
                        new_p = perturb_probs(orig_probs, intensity, preserve_rank=False)
                    new_probs.append({st: p for st, p in zip(change_at_states, new_p)})
                new_freq[s] = new_probs
        print('gradual drift observation space',new_freq)
        return new_freq



def action_penalty_dict(add_action):
        numadd = len(add_action)
        penalty = np.random.randint(-5.0,0,numadd)
        return {s: t for s, t in zip(add_action, penalty)}
        
def syn_state_action(prefix,n):
        return [f"{prefix}{i}" for i in range(n)]



def create_transition(states): 
        """
        create new transition matrix with the adde actions
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

def change_states_severity(self,add_states):
        #暂时不考虑add state的情况
        add_num = len(add_states)
        add_severity = np.random.randint(-5.0,5.0,add_num) #define severity with in range -5.0 - 5.0
        severity_dict = {s: t for s, t in zip(add_states, add_severity)}
        self.severity.update(**severity_dict)
        return