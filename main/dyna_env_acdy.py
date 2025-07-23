from typing import List, Tuple, Dict, Optional
import ast
import numpy as np
import pandas as pd
import gymnasium 
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import random
from itertools import repeat


"""在choose action 只从mask=True,动作有效，在这些动作中随机选择

在利用阶段自动忽略mask=False的动作（因为它们的Q值被设为-inf）

环境会在执行前验证动作有效性"""

class TaskEnv_actionD(gymnasium.Env):
    def __init__(self,
                 time_out: int = 10, #6 这个值一般怎么设置
                 timeout_reward=-1,
                 goal_reward=1,
                 time_reward_multiplicator=1,
                 max_actions = 20):
        self.flag = False #drift indicator
        self.severity = {'va': 0.0,
                         'po': -1.0,
                         'sib': -3.0,
                         'pp' : -4.0,
                         'Tau': 1.0}
        
        self.max_actions = max_actions
        self.motions = ['contact beeindigd/weggegaan',
          'client toegesproken/gesprek met client',
          'geen',
          'client afgeleid',
          'naar andere kamer/ruimte gestuurd',
          'met kracht tegen- of vastgehouden',
          'afzondering (deur op slot)']

        self.states = ['va','sib','pp','po','Tau'] 
        self.action_penalty_d = None
        self.disabled_actions = set()  # Use set for faster lookup - stores action names
        
        frequencies = pd.read_csv("frequencies.csv", index_col=0)
        for label in frequencies:
            for action in self.motions:
                frequencies[label][action] = ast.literal_eval(frequencies[label][action])
                
        self.goal = ['Tau']
        self.time_out = time_out
        self.timer = 0
        self.timeout_reward = timeout_reward
        self.goal_reward = goal_reward
        self.time_reward_multiplicator = time_reward_multiplicator
        self.seed(2)
        self.observation_space = frequencies
        self.observation_space_drift = None

        self.action_space = Discrete(self.max_actions) #max action space
        self.choice_positions =['va','pp','po','sib']
        self.positions = np.random.choice(self.choice_positions, size=1)
        self.episode_actions = []
        self.drift_type = None
        self.drift_swap = False
         
                # 初始化时所有新增动作槽位为None
        self.motions.extend([None] * (self.max_actions - len(self.motions)))
        

    def get_action_mask(self) -> List[bool]:
        """返回当前动作掩码，True表示有效动作，False表示无效动作"""
        mask = []
        for motion in self.motions:
            if motion is None:
                # 未分配的槽位永远无效
                mask.append(False)
            else:
                # 已分配的动作根据是否被禁用决定有效性
                mask.append(motion not in self.disabled_actions)
        return mask
    #new added action by default is valid
    

    """def get_valid_actions(self) -> List[int]:
        #返回有效动作的索引列表
        return [i for i, motion in enumerate(self.motions) 
               if motion is not None and motion not in self.disabled_actions]"""
    
    
    def get_action_info(self):
        """获取当前动作空间的完整信息"""
        valid_actions = self.get_valid_actions()
        disabled_names = [self.motions[i] for i in range(len(self.motions)) 
                      if self.motions[i] is not None and i not in valid_actions]
        
        return {
            'total_actions': self.max_actions,  # 最大动作空间
            'valid_actions': len(valid_actions),  # 有效动作数量
            'disabled_actions': len(disabled_names),
            'disabled_action_names': disabled_names,
            'action_mask': self.get_action_mask(),
            'defined_motions': [a for a in self.motions if a is not None]  # 实际定义的动作
        }
    
    def get_valid_actions(self) -> List[int]:
        """返回有效动作的索引列表"""
        return [i for i in range(len(self.motions)) 
               if self.motions[i] is not None 
               and self.motions[i] not in self.disabled_actions]
    

    def step(self, action: int, info=True):
        # 检查动作是否有效
        if action >= len(self.motions) or self.motions[action] is None:
            raise ValueError(f"Invalid action index: {action}")
            
        if not self.get_action_mask()[action]:
            raise ValueError(f"Attempted to take disabled action {action} ({self.motions[action]})")
        
        motion = self.motions[action]
        current_position = self.positions
        
        observation_space = self.observation_space
        if self.drift_type == 'gradual':
            if self.drift_swap == False:
                observation_space = random.choice([self.observation_space, self.observation_space_drift])
            else:
                observation_space = self.observation_space_drift
        
        #print(observation_space)

        new_position = self.get_next_state(observation_space[current_position][motion])[0]
        transition_reward = self.severity[new_position]

        if self.flag == False:
            action_penalty = self.get_action_reward(motion)
        else:
            action_penalty = self.get_action_reward(motion, self.action_penalty_d)

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
        self.timer = 0
        self.positions = np.random.choice(self.choice_positions)
        self.episode_actions = []
        return self.positions
                                          
    def _is_timeout(self) -> bool:
        return self.timer >= self.time_out

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
       
    def get_next_state(self, state_action):
        #print(state_action)
        states = []
        prob = []
        for key, value in state_action.items():
            states.append(key)
            prob.append(value)

        p = np.asarray(prob).astype('float64')
        p /= p.sum()
        return np.random.choice(states, size=1, p=p)
    
    def get_action_reward(self, action, add_penalty_dict=None):
        penalty = 0
        
        # Check if action is disabled
        if action in self.disabled_actions:
            raise ValueError(f"Attempted to use disabled action: {action}")
            
        if action == 'met kracht tegen- of vastgehouden' or action == 'afzondering (deur op slot)':
            penalty = -2
        elif action == 'client afgeleid' or action == 'contact beeindigd/weggegaan' or action == 'naar andere kamer/ruimte gestuurd':
            penalty = -1
        if add_penalty_dict is not None:
            penalty += add_penalty_dict.get(action, 0)
        return penalty

    def set_flag(self):
        self.flag = True
        return 

    def drift(self, add_actions=0, change_at_states=['pp','po','va','sib'], drift_dis_type='random', 
            intensity=0.5, drift_type='sudden', disable_actions: Optional[List[str]] = None):
        if not self.flag:
            return
            
        self.drift_type = drift_type
        
        # 处理动作增减
        if add_actions != 0:
            if add_actions > 0:
                # 添加新动作
                new_actions = self.syn_state_action('a', add_actions)
                #self.motions.extend(new_actions)
                self.motions[7:9] = new_actions #将7，8 index的元素none替换成新的action
                
                # 确保observation_space包含新动作（关键修改）
                for action in new_actions:
                    if action not in self.observation_space.index:
                        # 为新动作添加一行，用None初始化
                        self.observation_space.loc[action] = None
                
                # 为新动作在指定状态生成转移概率
                for state in change_at_states:
                    for action in new_actions:
                        self.observation_space.at[action, state] = self.drift_transition(self.states)
                
                # 为新动作分配随机惩罚
                self.action_penalty_d = self.action_penalty_dict(new_actions)
                print(f"Added {add_actions} new actions: {new_actions}")
                    
            elif add_actions < 0:
                if disable_actions is None:
                    available_actions = [a for a in self.motions if a not in self.disabled_actions]
                    num_to_disable = min(abs(add_actions), len(available_actions))
                    disable_actions = random.sample(available_actions, num_to_disable)
                
                self.disabled_actions.update(disable_actions)
                print(f"Disabled actions: {disable_actions}")
 
        """
        # Update observation space with all current motions
        if drift_type == 'sudden':
            self.observation_space = self.change_frequencies(
                self.motions,  # Pass all current motions
                change_at_states,
                original_freq=self.observation_space,
                drift_dis_type='random' if add_actions != 0 else drift_dis_type,
                intensity=intensity
            )
        """
        return
            
    def enable_action(self, action_name: str):
        """Re-enable a previously disabled action"""
        if action_name in self.disabled_actions:
            self.disabled_actions.remove(action_name)
            print(f"Enabled action: {action_name}")
        
    def disable_action(self, action_name: str):
        """Disable a specific action"""
        if action_name in self.motions:
            self.disabled_actions.add(action_name)
            print(f"Disabled action: {action_name}")
    
    """def get_action_info(self):
        #Get information about current action state
        total_actions = len(self.motions)
        valid_actions = len(self.get_valid_actions())
        disabled_actions = len(self.disabled_actions)
        
        return {
            'total_actions': total_actions,
            'valid_actions': valid_actions,
            'disabled_actions': disabled_actions,
            'disabled_action_names': list(self.disabled_actions),
            'action_mask': self.get_action_mask()
        }"""


# Helper functions remain the same as in original code
    def drift_transition(self,states):
        """
        Drift in transition probabilities: reassign probabilities among `states`,
        force the highest probability to bind with 'Tau', while keeping dict length unchanged.
        """
        np.random.seed(2)  # make results reproducible
        possible_path = len(states) #intotal 5 avaliable state
        path_num = random.randint(3, possible_path)
        
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

        return trans_dict
    def action_penalty_dict(self, add_action):
        numadd = len(add_action)
        penalty = np.random.randint(-5.0, 0, numadd)
        return {s: t for s, t in zip(add_action, penalty)}
            
    def syn_state_action(self, prefix, n):
        return [f"{prefix}{i}" for i in range(n)]

"""
    def change_frequencies(self, actions, change_at_states, original_freq=None, 
                        drift_dis_type='random', intensity=0.5):
        np.random.seed(2)
        if original_freq is None:
            raise ValueError("Original frequencies must be provided")
        
        # Create a new DataFrame with all current actions
        new_freq = pd.DataFrame(index=actions)
        
        for s in change_at_states:
            state_transitions = {}
            # Process existing actions
            for a in actions:
                if a in original_freq.columns:  # Existing action
                    if drift_dis_type == "random":
                        new_freq[s] = list(map(self.drift_transition, repeat(self.states, len(actions))))
                    else:
                        orig_probs = original_freq[s][a]
                        orig_probs = list(orig_probs.values()) if isinstance(orig_probs, dict) else orig_probs
                        if drift_dis_type == "similar":
                            state_transitions[a] = self.perturb_probs(orig_probs, intensity, True)
                        elif drift_dis_type == "inverse":
                            state_transitions[a] = self.perturb_probs(orig_probs, intensity, False)
                else:  # handel with  New action
                    new_freq[s] = list(map(self.drift_transition, repeat(self.states, len(actions))))
                # print(f"Added transition for new action {a} in state {s}")
            
            # Convert to Series
            new_freq[s] = pd.Series(state_transitions)
        
        return new_freq
"""
