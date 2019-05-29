from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            q_value =self.forward(state)
            action = q_value.max(1)[1].data[0]
            ######## YOUR CODE HERE! ########
        else:
            action = random.randrange(self.env.action_space.n)
        return action
        
def compute_td_loss(model, batch_size, gamma, replay_buffer):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = Variable(torch.FloatTensor(np.float32(states)))
    next_states = Variable(torch.FloatTensor(np.float32(next_states)), requires_grad=True)
    actions = Variable(torch.LongTensor(actions))
    rewards = Variable(torch.FloatTensor(rewards))
    done = Variable(torch.FloatTensor(dones))

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    q_values = model(states)
    next_q_values = model(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1-done)
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    ######## YOUR CODE HERE! ########
    return loss

def plot(frame_idx, rewards, losses, plot_dir):
    plt.figure(figsize=(20,4))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(plot_dir+'/%s-%s.png' % (frame_idx, np.mean(rewards[-10:])))
    plt.close()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".
        # If you are not familiar with the "deque" python library, please google it.
        # state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        indices = np.random.choice(len(self.buffer)-1, batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        ######## YOUR CODE HERE! ########

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def __len__(self):
        return len(self.buffer)