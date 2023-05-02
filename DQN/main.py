import numpy as np
import torch
import gym
from memory import *
from dqn import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from config import *


def main():
    env = gym.make('CartPole-v1')
    state,_ = env.reset()
    memory = Memory(NUM_STEP,1,env.observation_space.shape[0])    
    agent = DQN(2,env.observation_space.shape[0],GAMMA,EPSILON)
    rewardEpi = np.zeros(NUM_EPI)
    done = False
    #train
    
    for i in tqdm(range(NUM_EPI)):
        for _ in range(NUM_STEP):
            action = agent.sample_action(state)
            next_state, reward, terminate,truncate,_ = env.step(action)
            done = (terminate or truncate)
            rewardEpi[i] += reward
            memory.add(action,state,reward,next_state,done)

            act, stat, rew, next_stat, done = memory.randomSample()
            agent.update_parameter(stat,rew,next_stat,done)

            if(done):
                break
        
        state, _= env.reset()
        memory.reset()
        done = False

        if((i % INTERVAL) == 0):
            print(f'\n {i}:reward max:{max(rewardEpi[i-INTERVAL:], default=None)}, mean:{np.mean(rewardEpi[i-INTERVAL:])}, min:{min(rewardEpi[i-INTERVAL:])}')

    env.close()

    torch.save(agent.model,DQN_PARAM_PATH)

    fig = plt.figure()
    x = np.arange(len(rewardEpi))
    ax = fig.add_subplot(111)
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.plot(x,rewardEpi)
    plt.savefig('rewards')
    plt.show()
        
    

if __name__=="__main__":
    main()