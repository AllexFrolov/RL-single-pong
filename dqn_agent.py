from collections import deque, namedtuple
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from IPython.display import clear_output

moving_average = lambda x, **kw: pd.DataFrame({'x': np.asarray(x)}).x.ewm(**kw).mean().values


class DQNAgent:
    def __init__(self, model, target_model, loss_func, action_space, memory, n_multi_step=4,
                 transform=None, device='cpu', double_dqn=False, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05):
        self.device = device
        self.double_DQN = double_dqn
        self.n_multi_step = n_multi_step
        self.transform = transform
        self.model = model
        self.h = self.model.h
        self.w = self.model.w
        self.loss_func = loss_func
        self.target_model = target_model
        self.action_space = action_space
        self.memory = deque(maxlen=memory)
        self.gamma = torch.as_tensor([gamma], dtype=torch.float16, device=self.device)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.step = namedtuple('Step', ['state', 'action', 'reward', 'next_state', 'done'], rename=False)
        self.n_iter = 0
        self.loss_history = deque(maxlen=memory)
        self.rewards_history = []
        self.mean_loss_history = []

    def fit(self, env, steps, optimizer, start_train_steps=20, batch_size=200, train_every=10,
            update_model=10):
        self.optimizer = optimizer
        # mean_loss_history = []
        # loss_history = []
        # rewards_history = []
        for step in range(steps):
            state = env.reset()
            if self.transform:
                state = torch.as_tensor(state, dtype=torch.uint8)
                state = self.transform(state)
            done = False
            rewards = 0
            losses = []
            self.model.train(False)
            while not done:
                with torch.no_grad():
                    action = self.policy(state)
                    next_state, reward, done, _ = env.step(action)
                    if self.transform:
                        next_state = self.transform(torch.as_tensor(next_state, dtype=torch.uint8))
                    rewards += reward
                    self.memory.append(self.step(state, action, reward, next_state, done))
                    state = next_state
                    self.n_iter += 1
                    if self.n_iter % train_every == 0 and self.n_iter >= start_train_steps:
                        if len(self.memory) > batch_size:
                            batch = self.batch_create(batch_size)
                            losses.append(self.train(batch))
                    else:
                        losses.append(0)

                    if self.n_iter % update_model == 0 and self.n_iter >= start_train_steps:
                        self.target_model.set_parameters(self.model.get_parameters())

            self.rewards_history.append(rewards)
            self.mean_loss_history.append(np.mean(losses))
            self.loss_history += losses
            print(f'Step: {step}, rewards: {rewards:.0f}, mean loss: {np.mean(losses):.3f}')

            # Step epsilon
            if self.epsilon > self.epsilon_end:
                self.epsilon -= 0.01
                if self.epsilon < self.epsilon_end:
                    self.epsilon = self.epsilon_end

            # draw graph
            if (step + 1) % 10 == 0:
                clear_output(True)
                f, ax = plt.subplots(nrows=3, ncols=1, figsize=[10, 15])
                ax[0].set_title('Returns')
                ax[0].grid()
                ax[0].scatter(np.arange(len(self.rewards_history)), self.rewards_history, alpha=0.1)
                ax[0].plot(moving_average(self.rewards_history, span=10, min_periods=10))
                ax[1].set_title('Mean Loss')
                ax[1].grid()
                ax[1].plot(moving_average(self.mean_loss_history, span=10, min_periods=10), color='r')
                ax[2].set_title('Loss')
                ax[2].grid()
                ax[2].plot(np.arange(len(self.loss_history)), self.loss_history, color='b')
                plt.show()
        env.stop()

    def batch_create(self, batch_size):
        """
        Sample batch_size memories from the memory.
        NB: It deals the N-step DQN
        """
        # randomly pick batch_size elements from the buffer
        indices = np.random.choice(len(self.memory), batch_size, replace=False)

        states = []
        actions = []
        next_states = []
        rewards = []
        terminate = []  # don't used

        # for each indices
        for i in indices:
            sum_reward = 0
            states_look_ahead = self.memory[i].next_state
            done_look_ahead = self.memory[i].done

            # N-step look ahead loop to compute the reward and pick the new 'next_state' (of the n-th state)
            if not done_look_ahead:
                for n in range(self.n_multi_step):
                    if len(self.memory) > i + n:
                        # compute the n-th reward
                        sum_reward += (self.gamma ** n) * self.memory[i + n].reward
                        states_look_ahead = self.memory[i + n].next_state
                        if self.memory[i + n].done:
                            done_look_ahead = True
                            break
                        else:
                            done_look_ahead = False

            # Populate the arrays with the next_state, reward and terminate just computed
            states.append(self.memory[i].state)
            actions.append(self.memory[i].action)
            next_states.append(states_look_ahead)
            rewards.append(sum_reward)
            terminate.append(done_look_ahead)

        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64),
                np.array(next_states, dtype=np.float32),
                np.array(rewards, dtype=np.float32), np.array(terminate, dtype=np.uint8)
                )

    def train(self, batch):
        self.model.train(True)
        self.target_model.train(False)
        states, actions, next_states, rewards, terminate = batch

        # convert the data in tensors
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminate_t = torch.as_tensor(terminate, dtype=torch.bool, device=self.device)

        # Value of the action taken previously (recorded in actions_v) in the state_t
        with torch.enable_grad():
            if self.double_DQN:
                aq_next_value = torch.argmax(self.model(next_states_t), dim=-1).detach()
                next_state_values = (torch
                                     .gather(self.target_model(next_states_t), 1, aq_next_value.view(-1, 1))
                                     .squeeze(-1)
                                     .detach()
                                     )

            else:
                next_state_values = torch.max(self.target_model(next_states_t), dim=-1)[0]

            state_action_values = self.model(states_t).gather(1, actions_t.view(-1, 1)).squeeze(-1)
            next_state_values = next_state_values * (~terminate_t)
            expected_state_action_values = rewards_t + (self.gamma ** self.n_multi_step) * next_state_values
            loss = self.loss_func(expected_state_action_values, state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            if loss.item() > 1:
                return 1
        return loss.item()

    def policy(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            act = random.randint(0, self.action_space - 1)
            return act
        else:
            # self.model.train(False)
            q_value = self.model(torch.as_tensor(state, device=self.device))
            _, act = torch.max(q_value, dim=1)
            return int(act.item())
