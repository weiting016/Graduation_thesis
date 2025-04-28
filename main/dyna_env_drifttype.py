from typing import List, Tuple
import ast
import numpy as np
import pandas as pd
import gymnasium 
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import random
from itertools import repeat

class TaskEnv_driftype(gymnasium.Env):
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

    def drift(self, add_actions=0, change_at_states=None, drift_dis_type='random', intensity=0.5,drift_type='sudden'):
        """
        change of the env: the observation space: add actions and states
        assign new transition probability
        """
        if self.flag == True:
            # state action总数不变，只是部分path关闭
            print("drift happen")
            if add_actions != 0:
                syn_actions = syn_state_action('a', add_actions)  # ['a1','a2','a3']
                print(syn_actions)
                self.motions.extend(syn_actions)
                self.action_penalty_d = action_penalty_dict(syn_actions)
            
            #全局函数
            global change_frequencies

                # Apply drift based on type
            if drift_type == "sudden":
                # Sudden drift: full change at once
                self.observation_space = change_frequencies(
                    self.motions, 
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )
            
            elif drift_type == "gradual":
                # Gradual drift: interpolate between old and new
                if not hasattr(self, 'original_observation_space'):
                    self.original_observation_space = self.observation_space.copy()
                
                target_freq = change_frequencies(
                    self.motions,
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )
                # Linear interpolation (alpha = intensity)
                alpha = intensity
                mixed = pd.DataFrame(index=self.motions)
                for state in change_at_states:
                    mixed[state] = [
                        {
                            s: (1 - alpha) * self.original_observation_space[state][action][s] + 
                            alpha * target_freq[state][action][s]
                            for s in change_at_states
                        }
                        for action in self.motions
                    ]
                self.observation_space = mixed
            
            elif drift_type == "incremental":
                # Incremental drift: small random adjustments
                drifted = self.observation_space.copy()
                for state in change_at_states:
                    for action in self.motions:
                        dist = drifted[state][action]
                        for s in dist:
                            noise = np.random.normal(0, intensity * 0.1)  # Small step
                            dist[s] = max(dist[s] + noise, 0)
                        # Normalize
                        total = sum(dist.values())
                        if total > 0:
                            for s in dist:
                                dist[s] /= total
                        drifted[state][action] = dist
                self.observation_space = drifted
            
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

def perturb_probs(orig_probs, intensity, preserve_rank=True):
            """
            Helper function to generate perturbed probabilities.
            
            Args:
                orig_probs (list): Original probability distribution.
                intensity (float): Strength of perturbation.
                preserve_rank (bool): Whether to maintain original rank order.
            
            Returns:
                list: Perturbed probabilities (normalized).
            """
            orig_probs = np.array(orig_probs)
            # 确保输入概率有效
            if len(orig_probs) == 0 or not np.all(orig_probs >= 0):
                return list(orig_probs)  # 返回原始概率作为保底
            # 生成噪声
            noise = np.random.dirichlet(np.ones_like(orig_probs)) * intensity
            # 根据条件计算新概率
            if preserve_rank:
                # 保持原始排名顺序
                new_p = orig_probs * (1 - intensity) + noise
            else:
                # 完全重新分配概率
                new_p = noise
            # 确保概率有效
            new_p = np.abs(new_p)
            total = new_p.sum()
            # 避免除以零
            if total <= 0:
                return list[(np.ones_like(orig_probs) / len(orig_probs))] # 返回均匀分布
            new_p /= total  # 归一化
            print(new_p)
            return new_p.tolist()

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
                        new_p = new_p[::-1]  # Reverse order
                    new_probs.append({st: p for st, p in zip(change_at_states, new_p)})
                new_freq[s] = new_probs
        
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