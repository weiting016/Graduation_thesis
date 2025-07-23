import numpy as np
import random
from collections import defaultdict
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import deque
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import dyna_env_acdy
from dyna_env_acdy import TaskEnv_actionD


#value iteration agent
class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.states = env.states  # 假设状态空间固定
        self.actions = env.motions  # 所有可能动作
        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: None for s in self.states}

    def value_iteration(self):
       # print('run value iteration')
       # print('action mask:', self.env.get_action_mask())
        while True:
            delta = 0
            for s in self.states:
                if s == 'Tau':  # 终止状态
                    continue
                q_values=[]
                max_q = float('-inf')
             
                for a in self.actions:
                    if not self.env.get_action_mask()[self.actions.index(a)]:
                        #print(f"Skipping illegal action {a}") 
                        q_values.append(float('-inf'))
                        continue 

                    q = 0
                    for next_state, prob in self.env.observation_space[s][a].items():
                        #print(self.env.observation_space[s][a].items())
                        reward = self.env.severity.get(next_state, 0)+ self.env.get_action_reward(a)                    
                        if next_state == 'Tau':
                            q+= prob*reward
                        else:
                            q += prob * (reward + self.gamma * self.V[next_state])
                        if q > max_q:
                            max_q = q
                
                    q_values.append(q)

                max_q = max(q_values)
                delta = max(delta, abs(self.V[s] - max_q))
                self.V[s] = max_q
                self.policy[s] = np.argmax(q_values)
                
            if delta < self.theta:
                break

    def select_action(self, state):
        return self.policy[state]
    

#random action agent

class RandomMaskedAgent:
    def __init__(self, env):
        self.env = env
        self.actions = env.motions  # e.g., ['a0', ..., 'a19']

    def select_action(self):
        mask = self.env.get_action_mask()  # Boolean list of size 20
        available_actions = [a for a, m in zip(self.actions, mask) if m]
        
        if not available_actions:
            return None  # No valid actions available
        
        action = random.choice(available_actions)
        actionindex = self.actions.index(action)
        return actionindex

#masked q learn agent
class MaskedQLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        #max action space = 20
        #self.q_table = defaultdict(lambda: np.zeros(20))
        self.q_table = defaultdict(lambda: np.zeros(120))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions
    
    def apply_action_mask(self, q_values, action_mask):
        """Apply action mask to Q-values by setting invalid actions to -inf"""
        masked_q_values = q_values.copy()
        masked_q_values[~np.array(action_mask)] = -np.inf
        return masked_q_values
    
    def select_action(self, state, action_mask):
        """Select action using epsilon-greedy with action masking"""
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")
        
        # Epsilon-greedy with masking
        if np.random.random() < self.epsilon:
            # Random selection from valid actions only
            return np.random.choice(valid_actions)
        else:
            # Greedy selection from masked Q-values
            q_values = self.q_table[state]
            masked_q_values = self.apply_action_mask(q_values, action_mask)
            
            # Handle case where all Q-values are -inf (no valid actions were ever updated)
            if np.all(np.isinf(masked_q_values)):
                return np.random.choice(valid_actions)
            
            return np.argmax(masked_q_values)
    
    def update(self, state, action, reward, next_state, next_action_mask, done):
        """Update Q-table with action masking for next state"""
        if done:
            target = reward
        else:
            # Apply mask to next state Q-values for max calculation
            next_q_values = self.q_table[next_state]
            masked_next_q = self.apply_action_mask(next_q_values, next_action_mask)
            
            # Handle case where no valid actions in next state
            if np.all(np.isinf(masked_next_q)):
                target = reward  # No future reward if no valid actions
            else:
                target = reward + self.gamma * np.max(masked_next_q)
        
        # Standard Q-learning update
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])
    
    def get_policy(self, state, action_mask):
        """Get the current policy for a state considering action mask"""
        q_values = self.q_table[state]
        masked_q_values = self.apply_action_mask(q_values, action_mask)
        
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if len(valid_actions) == 0:
            return None
        
        if np.all(np.isinf(masked_q_values)):
            # If no Q-values updated yet, return random valid action
            return np.random.choice(valid_actions)
        
        return np.argmax(masked_q_values)

def train_with_drift(env, agent, episodes=1000, drift_episodes=None):
    """
    Train agent with potential drift in action space
    
    Args:
        env: Environment with action masking support
        agent: Q-learning agent
        episodes: Total training episodes
        drift_episodes: Episodes at which to introduce drift (list)
    
    """
    if drift_episodes is None:
        drift_episodes = []
    
    episode_rewards = []
    episode_lengths = []
    information = {}
    
    for episode in range(episodes):
        # Check if drift should occur
        state = env.reset()
        if episode in drift_episodes:
            print(f"\n=== Introducing Drift at Episode {episode} ===")
            env.set_flag()  # Enable drift
            
            # Example: Remove 2 actions by disabling them
            if episode == drift_episodes[0]:
                env.drift(add_actions=-2, 
                        drift_type='sudden',
                        
                        disable_actions=['client afgeleid', 'naar andere kamer/ruimte gestuurd'])
                
                print("Action info after drift:")
                print(env.get_action_info())

            if episode == drift_episodes[1]:
                        env.drift(add_actions=2, 
                        drift_type='sudden')
                       # disable_actions=['client afgeleid', 'naar andere kamer/ruimte gestuurd'])
                        print("Action info after drift:")
                        print(env.get_action_info())
        
        # Reset environment
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            steps += 1
            # Get current action mask
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
        
            
            # Take action
            try:
                next_state, reward, done, info = env.step(action)
            except ValueError as e:
                print(f"Error taking action: {e}")
                break

            if info != []:
                key = ''.join(str(x) for x in info)
                if key not in information:
                    information[key] = 1
                else:
                    information[key] += 1
 

            # Get mask for next state
            if not done:
                next_action_mask = env.get_action_mask()
            else:
                next_action_mask = [True] * len(env.motions)  # Doesn't matter for terminal state
            
            # Update Q-table
            agent.update(state, action, reward, next_state, next_action_mask, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        """
        if env.set_flag:
            with open('qlearn_action-drift.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(information)
        else:
                with open('qlearn_action-nodrift.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(information)
        """
    
    return episode_rewards, episode_lengths,information

def test_agent_performance(env, agent, episodes=1000):
    """Test agent performance with current action space"""
    test_rewards = []
    test_lengths = []
    
    # Temporarily disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
    
    # Restore exploration
    agent.epsilon = original_epsilon
    
    return np.mean(test_rewards), np.mean(test_lengths)




# implement  with action mask
class MaskedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(MaskedDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        if mask is not None:
            # 将无效动作的Q值设为负无穷，之后softmax时会变为0
            q_values = q_values.masked_fill(~mask, float('-inf'))
        return q_values
class MaskedDQNAgent:
    def __init__(self, env, state_dim=51, gamma=0.99, lr=1e-3, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=100)
        self.batch_size = 4
        
        self.policy_net = MaskedDQN(state_dim, self.action_dim)
        self.target_net = MaskedDQN(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.update_target_every = 50
        self.steps_done = 0
        

    def get_state_representation(self, state):
        """将离散状态转换为5维one-hot编码"""
        state_index = self.env.states.index(state)
        one_hot = np.zeros(51)
        one_hot[state_index] = 1
        return torch.FloatTensor(one_hot)
    
    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.append((
            self.get_state_representation(state),
            action,
            reward,
            self.get_state_representation(next_state),
            done,
            torch.BoolTensor(mask)
        ))
    
    def act(self, state, mask, training=True):
        if training and random.random() < self.epsilon:
            # 在有效动作中随机探索
            valid_actions = [i for i, m in enumerate(mask) if m]
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = self.get_state_representation(state)
                mask_tensor = torch.BoolTensor(mask)
                q_values = self.policy_net(state_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
                return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        masks = torch.stack(masks)
        
        # 当前Q值
        current_q = self.policy_net(states, masks).gather(1, actions.unsqueeze(1))
        
        # 下一个状态的Q值（使用target网络）
        with torch.no_grad():
            next_q = self.target_net(next_states, masks).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        self.update_epsilon()
        
        return loss.item()
    

    #ppo with dynamic action
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

class PPOAgentMask:
    def __init__(self, env, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.env = env
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
        self.action_masks = []
        
    def state_to_tensor(self, state):
        """Convert state string to tensor representation"""
        # Simple one-hot encoding for states
        #state_mapping = {'va': 0, 'sib': 1, 'pp': 2, 'po': 3, 'Tau': 4}
        state_mapping = {state: idx for idx, state in enumerate(self.env.states)}
        state_vector = np.zeros(51)
        if state in state_mapping:
            state_vector[state_mapping[state]] = 1.0
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state, action_mask):
        """Select action using current policy with action masking"""
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            
        # Apply action mask by setting invalid actions to very negative values
        masked_logits = logits.clone()
        masked_logits[0][~action_mask_tensor.bool()] = -float('inf')
        
        # Create distribution and sample
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done, action_mask):
        """Store transition in memory"""
        self.states.append(self.state_to_tensor(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(torch.FloatTensor(action_mask).to(self.device))
    
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
        action_masks = torch.stack(self.action_masks)
        
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
            
            # Apply action masks
            masked_logits = logits.clone()
            masked_logits[~action_masks.bool()] = -float('inf')
            
            # Create distribution
            dist = Categorical(logits=masked_logits)
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
        self.action_masks.clear()

def visualize_training_results(episode_rewards, episode_lengths, valid_actions_count):
    """Visualize training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(len(episode_rewards))
    
    # Plot 1: Episode Rewards
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
    # Rolling average
    window_size = 50
    if len(episode_rewards) >= window_size:
        rolling_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
        ax1.plot(episodes, rolling_avg, color='red', linewidth=2, label=f'{window_size}-Episode Average')
    
    ax1.axvline(x=300, color='orange', linestyle='--', alpha=0.8, label='Disable 2 Actions')
    ax1.axvline(x=600, color='green', linestyle='--', alpha=0.8, label='Add 2 Actions')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2.plot(episodes, episode_lengths, alpha=0.6, color='purple')
    ax2.axvline(x=300, color='orange', linestyle='--', alpha=0.8)
    ax2.axvline(x=600, color='green', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Valid Actions Count
    ax3.plot(episodes, valid_actions_count, color='brown', linewidth=2)
    ax3.axvline(x=300, color='orange', linestyle='--', alpha=0.8)
    ax3.axvline(x=600, color='green', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Valid Actions')
    ax3.set_title('Valid Actions Count Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Analysis (Reward segments)
    segments = {
        'Initial (1-299)': episode_rewards[0:299],
        'After Disable (300-599)': episode_rewards[300:600] if len(episode_rewards) > 599 else episode_rewards[300:],
        'After Add (600-999)': episode_rewards[600:] if len(episode_rewards) > 599 else []
    }
    
    segment_means = []
    segment_names = []
    for name, rewards in segments.items():
        if rewards:
            segment_means.append(np.mean(rewards))
            segment_names.append(name)
    
    bars = ax4.bar(segment_names, segment_means, color=['blue', 'orange', 'green'])
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Average Reward by Phase')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, segment_means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    for name, rewards in segments.items():
        if rewards:
            print(f"{name}: Mean={np.mean(rewards):.2f}, Std={np.std(rewards):.2f}")



#maseked DQN
class MetaMaskedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(MetaMaskedDQN, self).__init__()
        # 主网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 元学习相关参数
        self.fast_weights = None  # 用于存储快速适应的权重
        self.meta_lr = 0.1  # 元学习率，用于内循环更新
        
    def forward(self, x, mask=None, params=None):
        if params is None:
            params = dict(self.named_parameters())
            
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
        q_values = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        
        if mask is not None:
            q_values = q_values.masked_fill(~mask, float('-inf'))
        return q_values
    
    def clone_state(self):
        """克隆当前网络状态，用于元学习的内循环"""
        self.fast_weights = {k: v.clone() for k, v in self.named_parameters()}
        
    def adapt(self, loss):
        """在内循环中执行一步梯度下降"""
        grads = torch.autograd.grad(loss, self.fast_weights.values(), create_graph=True)
        self.fast_weights = {k: v - self.meta_lr * g for (k, v), g in zip(self.fast_weights.items(), grads)}

class MetaMaskedDQNAgent:
    def __init__(self, env, state_dim=51, gamma=0.99, lr=1e-3, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 meta_batch_size=5, num_meta_updates=5):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 元学习相关参数
        self.meta_batch_size = meta_batch_size  # 每个任务采样的episode数量
        self.num_meta_updates = num_meta_updates  # 内循环更新次数
        
        # 使用两个网络：一个用于元学习（快速适应），一个用于常规学习
        self.meta_net = MetaMaskedDQN(state_dim, self.action_dim)
        self.policy_net = MetaMaskedDQN(state_dim, self.action_dim)
        self.target_net = MetaMaskedDQN(state_dim, self.action_dim)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.update_target_every = 100
        self.steps_done = 0
        self.memory = deque(maxlen=10000)
        self.batch_size = 4
        
    def get_state_representation(self, state):
        """将离散状态转换为5维one-hot编码"""
       # state_index = ['va','sib','pp','po','Tau'].index(state)
        state_index = self.env.states.index(state)
        one_hot = np.zeros(51)
        one_hot[state_index] = 1
        return torch.FloatTensor(one_hot)
    
    def remember(self, state, action, reward, next_state, done, mask):
        self.memory.append((
            self.get_state_representation(state),
            action,
            reward,
            self.get_state_representation(next_state),
            done,
            torch.BoolTensor(mask)
        ))
    
    def act(self, state, mask, training=True, fast_adapt=False):
        if training and random.random() < self.epsilon:
            valid_actions = [i for i, m in enumerate(mask) if m]
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = self.get_state_representation(state)
                mask_tensor = torch.BoolTensor(mask)
                
                if fast_adapt:
                    # 使用快速适应的权重进行决策
                    q_values = self.meta_net(state_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), 
                                           params=self.meta_net.fast_weights)
                else:
                    # 使用常规策略网络进行决策
                    q_values = self.policy_net(state_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
                
                return q_values.argmax().item()
    
    def meta_update(self):
        """执行元更新（外循环）"""
        if len(self.memory) < self.meta_batch_size:
            return
        
        # 1. 采样一批任务/episodes
        episodes = random.sample(self.memory, self.meta_batch_size)
        
        # 2. 初始化元梯度
        meta_loss = 0
        
        for episode in episodes:
            # 3. 克隆网络状态（内循环开始）
            self.meta_net.load_state_dict(self.policy_net.state_dict())
            self.meta_net.clone_state()
            
            # 4. 内循环适应（在单个episode上快速适应）
            states, actions, rewards, next_states, dones, masks = episode
            
            for _ in range(self.num_meta_updates):
                # 计算当前episode的损失
                current_q = self.meta_net(states.unsqueeze(0), masks.unsqueeze(0), 
                                        params=self.meta_net.fast_weights)
                current_q = current_q.gather(1, torch.LongTensor([actions]).unsqueeze(1))
                
                with torch.no_grad():
                    next_q = self.target_net(next_states.unsqueeze(0), masks.unsqueeze(0))
                    next_q = next_q.max(1)[0]
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                
                loss = F.mse_loss(current_q.squeeze(), target_q)
                
                # 在内循环中执行一步梯度下降
                self.meta_net.adapt(loss)
            
            # 5. 计算适应后的损失，用于元梯度
            adapted_q = self.meta_net(states.unsqueeze(0), masks.unsqueeze(0), 
                                    params=self.meta_net.fast_weights)
            adapted_q = adapted_q.gather(1, torch.LongTensor([actions]).unsqueeze(1))
            meta_loss += F.mse_loss(adapted_q.squeeze(), target_q)
        
        # 6. 外循环更新（元更新）
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        return meta_loss.item()
    
    def replay(self):
        """常规经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        masks = torch.stack(masks)
        
        current_q = self.policy_net(states, masks).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states, masks).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        self.update_epsilon()
        
        return loss.item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)