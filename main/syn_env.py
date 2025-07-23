import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import List, Dict, Optional
import random



from typing import List, Tuple
import ast
import numpy as np
import pandas as pd
import gymnasium 
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import random
from itertools import repeat

#test sudden drift with different transition type

class GeneratedENV(gymnasium.Env):
    def __init__(self,
                 time_out: int = 50,
                 timeout_reward=-1,
                 goal_reward=5,
                 time_reward_multiplicator=1):
        """Contructor for the TaskEnvironment

        Args:
            size (int, optional): The size of the maze. Defaults to 15.
            time_out (int, optional): Time to explore the maze before the game is over. Defaults to 100.
        """
        self.flag = False #drift indicator

        #定义一个函数assigned severity for state
        action_rewards_df = pd.read_csv(
            "action_rewards.csv" )
# 转为字典 {action: reward}
        action_rewards_dict = dict(zip(action_rewards_df["action"], action_rewards_df["reward"]))

        state_rewards_df = pd.read_csv(
            "state_rewards.csv")

        # 转为字典 {state: reward}
        state_rewards_dict = dict(zip(state_rewards_df["state"], state_rewards_df["reward"]))




        self.severity = state_rewards_dict# 必须是字典形式 different reward when reach different state， 51个state, 也是读csv
        self.action_reward = action_rewards_dict
        
        self.motions = [ f'A{i}' for i in range(1,101)] # 50 ge action start with A1
        self.states = [f'S{i}' for i in range(1,51)]
        self.states.append('Tau')


        self.action_penalty_d = None #record added action with its penalty in diction form
        frequencies = pd.read_csv("Generated_MDP.csv", index_col=0)
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
        self.observation_space_drift = None #store drifted distribution 
        self.action_space = Discrete(len(self.motions))


        self.choice_positions = self.states[:-1]
        self.positions = np.random.choice(self.choice_positions, size=1)
        self.episode_actions = []
        self.drift_type = None
        self.drift_swap = False


    def step(self, action: int, info = True):
       # print("execute step function")
        motion = self.motions[action]
        current_position = self.positions
        #introduce function here, t=drift, which MDP used
        observation_space = self.observation_space
        if self.drift_type == 'gradual':
            if self.drift_swap==False:
            #print('drift gradual happen',type(self.observation_space),type(self.observation_space_drift))
                observation_space = random.choice([self.observation_space,self.observation_space_drift])
            else:
                observation_space =  self.observation_space_drift
                
        new_position = self.get_next_state(observation_space[current_position][motion])[0]

        transition_reward = self.severity[new_position]
        if self.flag == False: #without drift
            action_penalty = self.get_action_reward(motion)
        else: #add drift, use extended action_reward function
            action_penalty = self.get_action_reward(motion,self.action_penalty_d)

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
        if done and info: #done = True是一个episode结束
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
        self.episode_actions = [] #每一个episode会重置 所以要在每个episode见储存
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
        penalty = self.action_reward[action]
        #100 action，也是读csv，字典直接根据action name索引对应的reward

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
        np.random.seed(2)
        if self.flag == True:
            # state action总数不变，不关闭path
            #print("drift happen")
            self.drift_type = drift_type
            if add_actions != 0:
    
                syn_actions = self.syn_state_action('a', add_actions)
                #print(syn_actions)
                self.motions.extend(syn_actions)
                self.action_penalty_d = self.action_penalty_dict(syn_actions)

                # Apply drift based on type
            if drift_type == "sudden":
                # Sudden drift: full change at once
                self.observation_space = self.change_frequencies(
                    self.motions, 
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )
            elif drift_type == 'gradual':
                 self.observation_space_drift = self.change_frequencies(
                    self.motions, 
                    change_at_states,
                    original_freq=self.observation_space,
                    drift_dis_type=drift_dis_type,
                    intensity=intensity
                )

    

    def drift_transition(self, states):
        """
        Drift in transition probabilities: reassign probabilities among `states`,
        force the highest probability to bind with 'Tau', while keeping dict length unchanged.
        """
        np.random.seed(2)  # make results reproducible
        possible_path = len(states)
        path_num = random.randint(20, possible_path)
      
       #path_num = possible_path #dont close path
        # Generate transition probabilities
        trans_prob = np.random.dirichlet(np.ones(path_num), size=1)[0]
        #trans_prob = np.random.dirichlet(np.ones(possible_path), size=1)[0]
        
        if path_num < possible_path:
            trans_prob = np.pad(trans_prob, (0, possible_path - path_num), mode='constant', constant_values=0)
        np.random.shuffle(trans_prob)

        trans_dict = {}

        states_list = list(states)
        probs_list = trans_prob.tolist()

        # 找到 Tau 的索引
        tau_idx = states_list.index('Tau')

        # 如果 Tau 的概率为 0，则设置一个较小值并在其他地方扣除相应概率
        if probs_list[tau_idx] == 0.0:
            epsilon = 0.1
            probs_list[tau_idx] = epsilon

            # 从其他非零项中扣除 epsilon（平均分摊）
            non_tau_indices = [i for i in range(len(probs_list)) if i != tau_idx and probs_list[i] > 0]
            deduct = epsilon / len(non_tau_indices)
            for i in non_tau_indices:
                probs_list[i] -= deduct

        # 重新构建字典
        trans_dict = {s: p for s, p in zip(states_list, probs_list)}

        #print('len of transition dict',len(trans_dict))
        return trans_dict


    def perturb_probs(self,prob_list,intensity=0.5, ranking=True):
        #intensity 0.5-1
 
    # Convert to numpy array and normalize to ensure it sums to 1
        probs = np.array(prob_list, dtype=float)
        probs = probs / probs.sum()
        
        n = len(probs)
    
    # If intensity is 0, return original distribution
        if intensity == 0:
            return probs
        
        # Get the sorted indices to understand ranking
        sorted_indices = np.argsort(probs)
        
        # Create target distribution based on ranking preference
        if ranking:
            # Maintain original ranking: highest prob gets most concentration
            target_weights = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                # Linear weighting: smallest gets 1, largest gets n
                target_weights[idx] = i + 1
        else:
            # Reverse ranking: lowest prob gets most concentration  
            target_weights = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                # Reverse linear weighting: largest gets 1, smallest gets n
                target_weights[idx] = n - i
        
        # Normalize target weights to create target distribution
        target_dist = target_weights / np.sum(target_weights)
        
        # Interpolate between original and target based on intensity
        modified_probs = (1-intensity )* probs + (intensity) * target_dist
        
        # Ensure final normalization
        modified_probs = modified_probs / np.sum(modified_probs)
        
        return modified_probs


    def change_frequencies(self,actions, change_at_states, original_freq=None, 
                            drift_dis_type='random', intensity=0.5):
            """
            Generates a new transition probability matrix with controlled drift behavior.
            Only modifies probabilities for specified states, leaving others unchanged.
            """
            #print('change_frequencies function running')
            np.random.seed(2)  # Maintain reproducibility  
            if original_freq is None:
                raise ValueError("Original frequencies must be provided")       
            new_freq = original_freq.copy(deep=True)

            #print MDP check how it looks like with the probability, visualize it 
            # length of the traces of episode , different paths 
            # flip the action probability, how its react still choose the optimal 
            #design the own MDP, keep the Tau, flip prob
            
            for s in change_at_states:
                if drift_dis_type == "random":
                    # Original behavior: fully random #这里原来是change_at_state 
                    for a in actions:
                        new_freq[s][a] = self.drift_transition(self.states)
                    #new_freq[s] = list(map(self.drift_transition, repeat(self.states, len(actions))))

                else:
                    # Controlled drift based on original probabilities
                    new_probs = []
                    for a in actions:
                        orig_probs = list(original_freq[s][a].values())
                        if drift_dis_type == "similar":
                            # Perturb probabilities while preserving rank
                            new_p = self.perturb_probs(orig_probs, intensity, ranking=True)
                        elif drift_dis_type == "inverse":
                            # Invert probability ranks
                            new_p = self.perturb_probs(orig_probs, intensity, ranking=False)
                        original_states = list(original_freq[s][a].keys())
                        new_probs.append({st: p for st, p in zip(original_states, new_p)})
                        #这里原来是changeatstate
                    new_freq[s] = new_probs

            #print('sudden drift preserve ranking observation space',new_freq)
            #print(type(new_freq))
            return new_freq



    def action_penalty_dict(self,add_action):
            np.random.seed(2)
            numadd = len(add_action)
            penalty = np.random.randint(-5.0,0,numadd)
            return {s: t for s, t in zip(add_action, penalty)}
            
    def syn_state_action(self,prefix,n):
            return [f"{prefix}{i}" for i in range(n)]


    def create_transition(self,states): 
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
