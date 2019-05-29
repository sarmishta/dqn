import os
from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math
import numpy as np
import torch
import torch.optim as optim
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer, plot


num_frames = 1600000
batch_size = 64
gamma = 0.97  # The discount factor 
replay_initial = 10000
replay_buffer_max = 100000
lr = 0.00001
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

cwd = os.getcwd()
dir_name = str(num_frames)+"-"+str(replay_initial)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

plot_dir = dir_name+"/plots"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

video_dir = dir_name+"/video"
if not os.path.exists(video_dir):
    os.mkdir(video_dir)

#log file
file_w = "dump_log.csv"
f = open(dir_name+"/"+file_w, 'w')
f.write('Frames,Losses,Mean Rewards+\n')

env_id = "PongNoFrameskip-v4"
env, env_to_wrap = make_atari(env_id, video_dir)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

replay_buffer = ReplayBuffer(replay_buffer_max)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
optimizer = optim.Adam(model.parameters(), lr=lr)
if USE_CUDA:
    model = model.cuda()


epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0


state = env.reset()
for frame_idx in range(1, num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))

        f.write('%d,%f,%f' % (frame_idx, np.mean(losses), np.mean(all_rewards[-10:]))+"\n")
        plot(frame_idx, all_rewards, losses, plot_dir)

f.close()
env.close()
env_to_wrap.close()
