from collections import deque
import random
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

from IPython.display import clear_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

moving_average = lambda x, **kw: pd.DataFrame({'x': np.asarray(x)}).x.ewm(**kw).mean().values


class DQNAgent:
    def __init__(self, model, optimizer, loss_func, action_space, memory, transform=None, gamma=0.99, epsilon=0.1):
        self.transform = transform
        self.model = model
        self.h = self.model.h
        self.w = self.model.w
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.target_model = model
        self.action_space = action_space
        self.memory = deque(maxlen=memory)
        self.gamma = torch.Tensor([gamma]).to(device)
        self.epsilon = epsilon

    def fit(self, env, steps, batch_size=200, train_every=10, update_model=10, no_op_max=30, file_path=False):
        if file_path:
            self.load()  # Upload memory
        rewards_history = []
        for step in range(steps):
            state = env.reset()
            if self.transform:
                state = self.transform(state)
            done = False
            rewards = 0
            no_op = 0  # Idle counter
            while not done:
                with torch.no_grad():
                    action = self.policy(state)
                    next_state, reward, done, _ = env.step(action)
                    rewards += reward
                    if self.transform:
                        next_state = self.transform(next_state)
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state

                    if action == 1:
                        no_op += 1
                        if no_op == no_op_max:
                            done = True  # Stop playing
                    else:
                        no_op = 0

            rewards_history.append(rewards)

            # Train model
            if (step + 1) % train_every:
                if len(self.memory) > batch_size:
                    batch = random.sample(self.memory, batch_size)
                    self.train(batch)
                else:
                    batch = self.memory
                    self.train(batch)
                self.model.train(False)

            # Update target model
            if (step + 1) % update_model:
                self.target_model.set_parameters(self.model.get_parameters())
            # draw graph
            if (step + 1) % 10:
                clear_output(True)
                plt.figure(figsize=[12, 6])
                plt.title('Returns')
                plt.grid()
                plt.scatter(np.arange(len(rewards_history)), rewards_history, alpha=0.1)
                plt.plot(moving_average(rewards_history, span=10, min_periods=10))
                plt.show()

        self.save()  # save memory

    def train(self, batch):
        self.model.train(True)
        batch_size = len(batch)
        state0_batch = torch.zeros((batch_size, 3, self.h, self.w))
        state1_batch = torch.zeros((batch_size, 3, self.h, self.w))
        reward_batch = []
        action_batch = []
        # terminal1_batch = []
        for ind, (state0, action, reward, state1, terminal1) in enumerate(batch):
            state0_batch[ind] = state0
            state1_batch[ind] = state1
            reward_batch.append(reward)
            action_batch.append(action)
            # terminal1_batch.append(0. if terminal1 else 1.)

        state0_batch = state0_batch.to(device)
        state1_batch = state1_batch.to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        # terminal1_batch = torch.Tensor(terminal1_batch).to(device)
        target_q_values1 = self.target_model(state1_batch).detach()

        q_values1, _ = torch.max(target_q_values1, dim=1)

        discounted_reward_batch = self.gamma * q_values1
        # discounted_reward_batch *= terminal1_batch
        rs = reward_batch + discounted_reward_batch
        with torch.autograd.set_grad_enabled(True):
            q_values0 = self.model(state0_batch)
            loss = self.loss_func(rs.reshape(-1, 1), q_values0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self):
        torch.save({'memory': self.memory,
                    'state_dict': self.model.state_dict}, 'DQNSave.dqn')

    def load(self):
        loader = torch.load('DQNSave.dqn')
        self.memory = loader['memory']
        self.model.state_dict = loader['state_dict']

    def policy(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            self.model.train(False)
            actions = self.model(torch.FloatTensor(state).to(device))
            return torch.argmax(F.softmax(actions), dim=-1).detach().cpu().numpy()
