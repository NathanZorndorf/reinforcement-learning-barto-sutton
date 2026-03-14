import gymnasium as gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop
import random
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import os, sys

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)


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
    plt.savefig(filename)
    plt.show()


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
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Predict Q-values for starting state
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Vectorized update: where done=True, use reward; else use reward + gamma*max_next_q
        max_next_q = np.max(next_q_values, axis=1)
        targets[np.arange(batch_size), actions] = rewards + self.gamma * max_next_q * (~dones)

        # Train on batch
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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

    # Save weights
    weights_file = 'two-networks/dqn_weights.h5'
    agent.model.save_weights(weights_file)

    # Create final video with greedy policy
    agent.epsilon = 0.00
    create_video(agent, env, filename="two-networks/cartpole_dqn_video_final.mp4")
