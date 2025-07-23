import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random

import numpy as np
from IPython import get_ipython
import random
import matplotlib.pyplot as plt
from IPython import display
from tqdm.notebook import tqdm
from typing import Tuple, List
import itertools as it
import pandas as pd
import statistics as s

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import ast
from torch.distributions import Categorical

import torch.nn.functional as F
from collections import deque, defaultdict
from tqdm import tqdm

import syn_env
from syn_env import GeneratedENV
env = GeneratedENV()



class OptimalPolicyAgent:
    def __init__(self, env: GeneratedENV ):
        self.policy = None
        self.state_value =None
        actionlist = env.motions
        self.actions = {a: i for i, a in enumerate(actionlist)}
        self.env = env
        
    def act(self, state):
        actionname = self.policy[state]
        return self.actions[actionname]

    def value_iteration(self, gamma=0.995, theta=1e-6):
        state_list = self.env.states
        action_list = self.env.motions
        state_idx = {s: i for i, s in enumerate(state_list)}
        
        n_states = len(state_list)

        V = np.zeros(n_states)
        policy = np.zeros(n_states, dtype=int)

        while True:
            delta = 0
            for s_name in state_list :
                if s_name =='Tau':
                    continue
                s = state_idx[s_name]
                q_values = []
                for a_name in action_list:
                    q = 0
                    for s_prime_name, prob in self.env.observation_space[s_name][a_name].items():
                        s_prime = state_idx[s_prime_name] #next state
                        reward = self.env.severity.get(s_prime_name, 0)+ self.env.get_action_reward(a_name)
                        if s_prime == 'Tau':
                            q+= prob*reward
                        else:
                            q += prob * (reward + gamma * V[s_prime])
                    q_values.append(q)

                max_q = max(q_values)
                delta = max(delta, abs(V[s] - max_q))
                V[s] = max_q
                policy[s] = np.argmax(q_values)
                
            if delta < theta:
                break

        
        optimal_policy = {state_list[s]: action_list[policy[s]] for s in range(n_states) if state_list[s]!='Tau'}

        self.policy = optimal_policy
        self.state_value = V


class RandomAgent:
    def __init__(self, env: GeneratedENV ):
        self.n_actions = len(env.motions)

    def act(self,state):
        return random.randint(0, self.n_actions - 1)


class Q_learning_Agent():
    def __init__(self,
                 env: GeneratedENV ,
                 exploration_rate: float = 0.2,
                 learning_rate: float = 0.1,
                 discount_factor: float = 1) -> int:
        self.env = env
        self.epsilon = exploration_rate  # A random agent "explores" always, so epsilon will be 1
        self.alpha = learning_rate  # A random agent never learns, so there's no need for a learning rate
        self.gamma = discount_factor  # A random agent does not update it's q-table. Hence, it's zero.
        self.q_table = np.zeros((50,100), dtype=float) 

        self.actions = env.action_space

    def select_action(self, state, use_greedy_strategy: bool = False) -> int:
        if not use_greedy_strategy:
            if random.random() < self.epsilon:
                next_action = self.actions.sample()
                return next_action

        x = self.env.observation_space.columns.to_list().index(state)
        max_val = np.max(self.q_table[x, :])
        find_max_val = np.where(self.q_table[x, :] == max_val)
        next_action = np.random.choice(find_max_val[0])
        return next_action

    def learn(self, state, action, next_state, reward, done):  
        x = self.env.observation_space.columns.to_list().index(state)
        next_max_val = 0
        if next_state != 'Tau':
           # print("length of observation space colummns",len(env.observation_space.columns))
            x_ = self.env.observation_space.columns.to_list().index(next_state)
            next_max_val = np.max(self.q_table[x_, :])
        # Update the value in the q-table using Q(S,A) <- Q(S,A) + alpha(R + gamma(max action value) - Q(S,A))
        self.q_table[x, action] += self.alpha * (reward + (self.gamma*next_max_val) - self.q_table[x, action])

    """
    # 定义 State-Action Embedding 和 Q-Network
class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim=16):
        super(StateActionEmbedding, self).__init__()
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, state, action):
        state_embed = self.state_embedding(state)
        action_embed = self.action_embedding(action)
        return torch.cat([state_embed, action_embed], dim=-1)

class QNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_action_embed):
        return self.fc(state_action_embed)

# 定义 DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, embedding_dim=16, gamma=0.9, lr=1e-3, epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.losses = []
        self.episode_rewards = []

        # 网络
        self.embedding_net = StateActionEmbedding(state_dim, action_dim, embedding_dim)
        self.q_net = QNetwork(embedding_dim)
        self.target_q_net = QNetwork(embedding_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(list(self.embedding_net.parameters()) + list(self.q_net.parameters()), lr=lr)

        # 经验回放
        self.memory = deque(maxlen=10)

    def get_action(self, state, state_to_index):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机动作
        else:
            state_index = state_to_index[state]  # 将 state 转换为索引
            state_one_hot = torch.zeros(self.state_dim).unsqueeze(0)
            state_one_hot[0, state_index] = 1  # 转换为 one-hot 编码
            q_values = []
            for action in range(self.action_dim):
                action_one_hot = torch.zeros(self.action_dim).unsqueeze(0)
                action_one_hot[0, action] = 1
                state_action_embed = self.embedding_net(state_one_hot, action_one_hot)
                q_value = self.q_net(state_action_embed)
                q_values.append(q_value.item())
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=1, state_to_index=None):
        if len(self.memory) < batch_size:
            return

        # 从经验回放中采样一个样本（batch_size=1）
        state, action, reward, next_state, done = self.memory[-1]

        # 将 state 和 next_state 转换为 one-hot 编码
        state_index = state_to_index[state]
        state_one_hot = torch.zeros(self.state_dim).unsqueeze(0)
        state_one_hot[0, state_index] = 1

        next_state_index = state_to_index[next_state]
        next_state_one_hot = torch.zeros(self.state_dim).unsqueeze(0)
        next_state_one_hot[0, next_state_index] = 1

        # 将 action 转换为 one-hot 编码
        action_one_hot = torch.zeros(self.action_dim).unsqueeze(0)
        action_one_hot[0, action] = 1

        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # 计算当前 Q 值
        state_action_embed = self.embedding_net(state_one_hot, action_one_hot)
        current_q = self.q_net(state_action_embed)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = []
            for next_action in range(self.action_dim):
                next_action_one_hot = torch.zeros(self.action_dim).unsqueeze(0)
                next_action_one_hot[0, next_action] = 1
                next_state_action_embed = self.embedding_net(next_state_one_hot, next_action_one_hot)
                next_q = self.target_q_net(next_state_action_embed)
                next_q_values.append(next_q.item())
            next_q_max = max(next_q_values)
            target_q = reward + (1 - done) * self.gamma * next_q_max

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 更新目标网络
    def plot_training_progress(self):
        #绘制训练过程中的reward变化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 绘制episode rewards
        ax1.plot(self.episode_rewards, label='Episode Reward', alpha=0.6)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制损失变化
        ax2.plot(self.losses, label='Loss', alpha=0.6)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
"""

#implement metaDQN

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MetaDQNAgent:
    def __init__(self, env):
        self.env = env
        
        # Initialize state and action encoders
        self.state_encoder = StateActionEncoder(env.states)
        self.action_encoder = StateActionEncoder(env.motions)
        
        self.state_size = len(env.states)
        self.action_size = len(env.motions)
        
        self.memory = deque(maxlen=500) #online, so keep only small batch of memory
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.2   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 4
        self.update_target_every = 50
        
        # Main network
        self.model = DQN(self.state_size, self.action_size)
        # Target network (for stability)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Meta-learning components
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.fast_weights = None
        self.meta_batch_size = 5
        self.meta_loss = None
        
        # Tracking
        self.rewards_history = []
        self.epsilons_history = []
        self.loss_history = []
        self.adaptation_scores = []
        
        # For action masking
        self.valid_actions_cache = {}

    def remember(self, state, action, reward, next_state, done):
        state_enc = self.state_encoder.encode(state)
        action_idx = self.action_encoder.encode(action)
        next_state_enc = self.state_encoder.encode(next_state)
        self.memory.append((state_enc, action_idx, reward, next_state_enc, done))
        

    
    def get_valid_actions(self, state):
        """Cache valid actions for each state to handle action masking"""
        if state not in self.valid_actions_cache:
            # Get all possible actions from environment
            self.valid_actions_cache[state] = list(range(len(self.env.motions)))
        return self.valid_actions_cache[state]
    
    def act(self, state, training=True):
        valid_actions = self.get_valid_actions(state)
        
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_enc = self.state_encoder.encode(state)
        state_enc = torch.FloatTensor(state_enc).unsqueeze(0)
        
        with torch.no_grad():
            action_values = self.model(state_enc)
        
        # Convert to numpy and mask invalid actions
        action_values = action_values.squeeze().numpy()
        masked_values = -np.inf * np.ones_like(action_values)
        masked_values[valid_actions] = action_values[valid_actions]
        
        return np.argmax(masked_values)
    

    def compute_loss(self, batch, model):
            # 解包批次数据
        states, actions, rewards, next_states, dones = zip(*batch)  
    # 转为张量并确保正确形状
        states = torch.FloatTensor(np.array(states))  # shape: [batch_size, state_dim]
        next_states = torch.FloatTensor(np.array(next_states))  # shape: [batch_size, state_dim] 
    # 关键修正：处理actions维度
        actions = torch.LongTensor(actions)  # 先转为1D张量
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # 转为[batch_size, 1]
        elif actions.dim() > 2:
            actions = actions.squeeze()  # 去除多余维度
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
    
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # [batch_size, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1)      # [batch_size, 1]
        self._validate_shapes(states, actions, rewards, next_states, dones)
        # 获取Q值
        q_values = model(states)  # [batch_size, action_size]
        
        # 收集实际采取动作的Q值
        current_q = q_values.gather(1, actions)  # [batch_size, 1]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        return F.mse_loss(current_q, target_q)

    
    def _validate_shapes(self, states, actions, rewards, next_states, dones):
        """验证所有张量的形状是否正确"""
        assert states.dim() == 2, f"States should be 2D, got {states.dim()}"
        assert actions.dim() == 2, f"Actions should be 2D, got {actions.dim()}"
        assert rewards.dim() == 2, f"Rewards should be 2D, got {rewards.dim()}"
        assert next_states.dim() == 2, f"Next states should be 2D, got {next_states.dim()}"
        assert dones.dim() == 2, f"Dones should be 2D, got {dones.dim()}"
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        loss = self.compute_loss(batch, self.model)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def meta_update(self):
        if len(self.memory) < self.meta_batch_size * self.batch_size:
            return 0
        
        # 保存原始权重
        original_weights = {k: v.clone() for k, v in self.model.named_parameters()}
        
        task_losses = []
        for _ in range(self.meta_batch_size):
            # 采样一个任务批次
            batch = random.sample(self.memory, self.batch_size)
            
            # 内循环：计算梯度并创建快速权重
            with torch.enable_grad():  # 确保计算梯度
                loss = self.compute_loss(batch, self.model)
                gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                
                # 创建快速权重（保持计算图连接）
                fast_weights = {
                    name: param - self.learning_rate * grad
                    for (name, param), grad in zip(self.model.named_parameters(), gradients)
                }
            
            # 计算快速权重下的损失（保持计算图）
            with torch.enable_grad():
                # 临时使用快速权重进行前向传播
                def fast_forward(x):
                    x = F.relu(F.linear(x, fast_weights['fc1.weight'], fast_weights['fc1.bias']))
                    x = F.relu(F.linear(x, fast_weights['fc2.weight'], fast_weights['fc2.bias']))
                    return F.linear(x, fast_weights['fc3.weight'], fast_weights['fc3.bias'])
                
                task_loss = self.compute_loss(batch, fast_forward)
                task_losses.append(task_loss)
        
        # 外循环：元更新
        self.meta_loss = torch.mean(torch.stack(task_losses))
        self.meta_optimizer.zero_grad()
        self.meta_loss.backward()
        self.meta_optimizer.step()
        
        #return meta_loss.item()
    
    def _transfer_weights(self, old_model, old_state_encoder, old_action_encoder):
        """Transfer weights from old model to new model, handling dimension changes"""
        # Create mapping from old to new indices
        state_mapping = self._create_mapping(old_state_encoder.vocab, self.state_encoder.vocab)
        action_mapping = self._create_mapping(old_action_encoder.vocab, self.action_encoder.vocab)
        
        # Transfer weights layer by layer
        with torch.no_grad():
            # FC1 layer (input is state)
            self._transfer_fc_layer(old_model.fc1, self.model.fc1, state_mapping, None)
            
            # FC2 layer (no dimension changes)
            if old_model.fc2.weight.shape == self.model.fc2.weight.shape:
                self.model.fc2.load_state_dict(old_model.fc2.state_dict())
            
            # FC3 layer (output is action)
            self._transfer_fc_layer(old_model.fc3, self.model.fc3, None, action_mapping)
    
    def _transfer_fc_layer(self, old_layer, new_layer, input_mapping, output_mapping):
        """Helper function to transfer weights for a fully connected layer"""
        old_weights = old_layer.weight.data
        old_bias = old_layer.bias.data if old_layer.bias is not None else None
        
        new_weights = torch.zeros_like(new_layer.weight.data)
        new_bias = torch.zeros_like(new_layer.bias.data) if new_layer.bias is not None else None
        
        # Handle input dimension mapping
        if input_mapping is not None:
            for new_in, old_in in input_mapping.items():
                new_weights[:, new_in] = old_weights[:, old_in]
        else:
            min_in = min(old_weights.shape[1], new_weights.shape[1])
            new_weights[:, :min_in] = old_weights[:, :min_in]
        
        # Handle output dimension mapping
        if output_mapping is not None:
            for new_out, old_out in output_mapping.items():
                new_weights[new_out, :] = old_weights[old_out, :]
                if new_bias is not None:
                    new_bias[new_out] = old_bias[old_out]
        else:
            min_out = min(old_weights.shape[0], new_weights.shape[0])
            new_weights[:min_out, :] = old_weights[:min_out, :]
            if new_bias is not None:
                new_bias[:min_out] = old_bias[:min_out]
        
        new_layer.weight.data = new_weights
        if new_bias is not None:
            new_layer.bias.data = new_bias
    
    def _create_mapping(self, old_vocab, new_vocab):
        """Create mapping from old to new indices for common elements"""
        mapping = {}
        for new_idx, item in enumerate(new_vocab):
            if item in old_vocab:
                old_idx = old_vocab.index(item)
                mapping[new_idx] = old_idx
        return mapping
    
    def evaluate_adaptation(self, num_episodes=10):
        """Evaluate agent's current performance"""
        total_rewards = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action_idx = self.act(state, training=False)
                next_state, reward, done, _ = self.env.step(action_idx)
                state = next_state
                episode_reward += reward
            
            total_rewards += episode_reward
        
        return total_rewards / num_episodes
    
    def plot_training(self, show_window=50):
        plt.figure(figsize=(15, 5))
        
        # 1. Plot rewards
        plt.subplot(1, 3, 1)
        
        # Ensure we have enough episodes for moving average
        if len(self.rewards_history) >= show_window:
            moving_avg = np.convolve(self.rewards_history, np.ones(show_window)/show_window, mode='valid')
            # Correct x-axis coordinates for moving average
            x_ma = np.arange(show_window-1, len(self.rewards_history))
            plt.plot(x_ma, moving_avg, 'r-', linewidth=2, label=f'MA({show_window})')
        
        # Plot raw rewards
        plt.plot(self.rewards_history, alpha=0.3, label='Raw')
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        # 2. Plot epsilon decay
        plt.subplot(1, 3, 2)
        plt.plot(self.epsilons_history)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # 3. Plot adaptation scores
        plt.subplot(1, 3, 3)
        if len(self.adaptation_scores) > 0:
            # Calculate correct x-axis positions for adaptation scores
            eval_every = len(self.rewards_history) / len(self.adaptation_scores)
            x = np.arange(0, len(self.rewards_history), eval_every)
            # Ensure we don't have more points than scores
            x = x[:len(self.adaptation_scores)]
            plt.plot(x, self.adaptation_scores, 'g-', marker='o')
            plt.title('Adaptation Scores')
            plt.xlabel('Episode')
            plt.ylabel('Avg Reward (eval)')
        
        plt.tight_layout()
        plt.show()
        
class StateActionEncoder:
    """Helper class to handle one-hot encoding of states and actions"""
    def __init__(self, items):
        self.vocab = sorted(items)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.vocab)}
    
    def encode(self, item):
        """Convert item to one-hot encoding"""
        if item not in self.item_to_idx:
            # Handle unknown items (shouldn't happen in our case)
            return np.zeros(len(self.vocab))
        encoding = np.zeros(len(self.vocab))
        encoding[self.item_to_idx[item]] = 1
        #print(self.item_to_idx[item])
        #print(encoding)
        return encoding
    
    def decode(self, encoding):
        """Convert one-hot encoding back to item"""
        idx = np.argmax(encoding)
        return self.vocab[idx]



class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgentnew:
    def __init__(self, env):
        self.env = env
        
        # Initialize state and action encoders
        self.state_encoder = StateActionEncoder(env.states)
        self.action_encoder = StateActionEncoder(env.motions)
        
        self.state_size = len(env.states)
        self.action_size = len(env.motions)
        
        self.memory = deque(maxlen=500) #online, so keep only small batch of memory
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.2   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 4
        self.update_target_every = 50
        
        # Main network
        self.model = DQN(self.state_size, self.action_size)
        # Target network (for stability)
        self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Tracking
        self.rewards_history = []
        self.epsilons_history = []
        self.loss_history = []
        
        # For action masking
        self.valid_actions_cache = {}

    def remember(self, state, action, reward, next_state, done):
        state_enc = self.state_encoder.encode(state)
        action_idx = self.action_encoder.encode(action)
        next_state_enc = self.state_encoder.encode(next_state)
        self.memory.append((state_enc, action_idx, reward, next_state_enc, done))
    
    def get_valid_actions(self, state):
        """Cache valid actions for each state to handle action masking"""
        if state not in self.valid_actions_cache:
            # Get all possible actions from environment
            self.valid_actions_cache[state] = list(range(len(self.env.motions)))
        return self.valid_actions_cache[state]
    
    def act(self, state, training=True):
        valid_actions = self.get_valid_actions(state)
        
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_enc = self.state_encoder.encode(state)
        state_enc = torch.FloatTensor(state_enc).unsqueeze(0)
        
        with torch.no_grad():
            action_values = self.model(state_enc)
        
        # Convert to numpy and mask invalid actions
        action_values = action_values.squeeze().numpy()
        masked_values = -np.inf * np.ones_like(action_values)
        masked_values[valid_actions] = action_values[valid_actions]
        
        return np.argmax(masked_values)
    
    def compute_loss(self, batch, model):
        # 解包批次数据
        states, actions, rewards, next_states, dones = zip(*batch)  
        # 转为张量并确保正确形状
        states = torch.FloatTensor(np.array(states))  # shape: [batch_size, state_dim]
        next_states = torch.FloatTensor(np.array(next_states))  # shape: [batch_size, state_dim] 
        # 关键修正：处理actions维度
        actions = torch.LongTensor(actions)  # 先转为1D张量
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # 转为[batch_size, 1]
        elif actions.dim() > 2:
            actions = actions.squeeze()  # 去除多余维度
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
    
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # [batch_size, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1)      # [batch_size, 1]
        self._validate_shapes(states, actions, rewards, next_states, dones)
        # 获取Q值
        q_values = model(states)  # [batch_size, action_size]
        
        # 收集实际采取动作的Q值
        current_q = q_values.gather(1, actions)  # [batch_size, 1]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        return F.mse_loss(current_q, target_q)
    
    def _validate_shapes(self, states, actions, rewards, next_states, dones):
        """验证所有张量的形状是否正确"""
        assert states.dim() == 2, f"States should be 2D, got {states.dim()}"
        assert actions.dim() == 2, f"Actions should be 2D, got {actions.dim()}"
        assert rewards.dim() == 2, f"Rewards should be 2D, got {rewards.dim()}"
        assert next_states.dim() == 2, f"Next states should be 2D, got {next_states.dim()}"
        assert dones.dim() == 2, f"Dones should be 2D, got {dones.dim()}"
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        loss = self.compute_loss(batch, self.model)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if len(self.loss_history) % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    
    def plot_training(self, show_window=50):
        plt.figure(figsize=(15, 5))
        
        # 1. Plot rewards
        plt.subplot(1, 2, 1)
        
        # Ensure we have enough episodes for moving average
        if len(self.rewards_history) >= show_window:
            moving_avg = np.convolve(self.rewards_history, np.ones(show_window)/show_window, mode='valid')
            # Correct x-axis coordinates for moving average
            x_ma = np.arange(show_window-1, len(self.rewards_history))
            plt.plot(x_ma, moving_avg, 'r-', linewidth=2, label=f'MA({show_window})')
        
        # Plot raw rewards
        plt.plot(self.rewards_history, alpha=0.3, label='Raw')
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        # 2. Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(self.epsilons_history)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()

class StateActionEncoder:
    """Helper class to handle one-hot encoding of states and actions"""
    def __init__(self, items):
        self.vocab = sorted(items)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.vocab)}
    
    def encode(self, item):
        """Convert item to one-hot encoding"""
        if item not in self.item_to_idx:
            # Handle unknown items (shouldn't happen in our case)
            return np.zeros(len(self.vocab))
        encoding = np.zeros(len(self.vocab))
        encoding[self.item_to_idx[item]] = 1
        return encoding
    
    def decode(self, encoding):
        """Convert one-hot encoding back to item"""
        idx = np.argmax(encoding)
        return self.vocab[idx]
    

#implemnt PPO
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        shared_out = self.shared(state)
        policy_logits = self.policy_head(shared_out)
        value = self.value_head(shared_out)
        return policy_logits, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        # Storage for rollouts
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def state_to_tensor(self, state):
        """Convert state string to tensor representation"""
        # One-hot encoding for 5 states
        state_mapping = {f'S{i}': i for i in range(1, 51)} 
        state_mapping['Tau'] = 51
        state_vector = np.zeros(51)
        if state in state_mapping:
            state_vector[state_mapping[state]] = 1.0
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            
        # Create distribution and sample
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.states.append(self.state_to_tensor(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        returns = []
        advantages = []
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i+1]
                next_val = self.values[i+1]
            
            delta = self.rewards[i] + self.gamma * next_val * next_non_terminal - self.values[i]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return returns, advantages
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return
        
        # Convert lists to tensors
        states = torch.stack(self.states)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Forward pass
            logits, values = self.network(states)
            
            # Create distribution
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

