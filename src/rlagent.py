import torch
import torch.optim as optim
import random
import os
from collections import deque
import torch.nn.functional as F
from rlmodel import DQN
from config import NUMBER_OF_OUTPUTS, MODEL_PATH, LOAD_MODEL

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95):
        # gpu vs cpu calc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        # discovery rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.batch_size = 32
        self.memory = deque(maxlen=1000)

        if LOAD_MODEL and os.path.exists(MODEL_PATH):
            self.load_model()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, NUMBER_OF_OUTPUTS - 1)

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights from model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def load_model(self):
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())  # Sync target model
        print(f"Model loaded from {MODEL_PATH}")