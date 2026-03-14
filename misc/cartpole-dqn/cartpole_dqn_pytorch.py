import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Use CPU (no GPU available)
device = torch.device('cpu')


def create_video(agent, env, filename="cartpole_dqn_video.mp4"):
    state, _ = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    frames = []
    for time in range(500):
        frames.append(env.render())
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = next_state
        if done:
            break

    imageio.mimsave(filename, frames, fps=30)


def plot_metrics(scores, epsilons, filename="metrics.png"):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(scores)
    plt.title('Score per Episode vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(2, 1, 2)
    plt.plot(epsilons)
    plt.title('Epsilon vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()


class DQNNetwork(nn.Module):
    """Neural network for DQN agent"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Reduced from 500K for efficiency
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration-exploitation trade-off
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # PyTorch models
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().numpy()[0])

    def replay(self, batch_size):
        """Train on a batch of experiences"""
        minibatch = random.sample(self.memory, batch_size)

        # Extract batch data
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        # Predict Q-values for starting state
        q_values = self.model(states_tensor)

        # Predict Q-values for next state (with target network)
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor)
            max_next_q = torch.max(next_q_values, dim=1)[0]

        # Compute target Q-values
        targets = q_values.clone().detach()
        # Where done=True, use reward; else use reward + gamma*max_next_q
        targets[torch.arange(batch_size), actions_tensor] = rewards_tensor + self.gamma * max_next_q * (~dones_tensor).float()

        # Train on batch
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 32
    max_steps_per_episode = 500

    # initialize metric arrays
    scores, epsilons = [], []

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

        for time in range(max_steps_per_episode):
            action = agent.act(state)

            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward

            if done:
                print("Episode: {}/{}, Total Reward: {}".format(episode + 1, episodes, total_reward))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # track metrics
        scores.append(total_reward)
        epsilons.append(agent.epsilon)

        # Plot every 50 episodes instead of every episode (more efficient)
        if episode % 50 == 0 and episode > 0:
            plot_metrics(scores, epsilons, filename="two-networks/cartpole_dqn_metrics.png")

        # update target network and create video every 100 episodes
        if episode % 100 == 0 and episode > 0:
            agent.update_target_model()
            create_video(agent, env, filename="two-networks/cartpole_dqn_video_{}.mp4".format(episode))

    # Save model
    weights_file = 'two-networks/dqn_model.pt'
    agent.save_model(weights_file)

    # Create final video with greedy policy
    agent.epsilon = 0.00
    create_video(agent, env, filename="two-networks/cartpole_dqn_video_final.mp4")
