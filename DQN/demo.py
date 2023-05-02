import numpy as np
import torch
from dqn import *
from matplotlib import pyplot as plt
import gym
from config import *
from tqdm import tqdm

env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
agent = DQN(2,env.observation_space.shape[0],GAMMA,EPSILON)
agent.model = torch.load(DQN_PARAM_PATH)
done = False
action_his = np.array([])

for i in tqdm(range(100)):
    while(not done):
        action = agent.sample_action(state)
        _, _ , ter, trun, _ = env.step(action)
        done = (ter or trun)
        action_his = np.append(action_his, action)

    state, _ = env.reset()
    done = False

fig = plt.figure()
x = np.arange(len(action_his))
ax = fig.add_subplot(111)
ax.set_xlabel('num')
ax.set_ylabel('action')
ax.plot(x,action_his)
plt.savefig('action_history.svg')
plt.show()
        

env.close()

