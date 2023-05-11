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
    memory = DequeMemory(MEMORY)    
    agent = DQN(2,env.observation_space.shape[0],GAMMA,EPSILON)
    rewardEpi = np.zeros(NUM_EPI)
    lossEpi =  np.zeros(NUM_EPI)
    done = False
    #train
    
    for i in tqdm(range(NUM_EPI)):
        for step in range(NUM_STEP):
            action = agent.sample_action(torch.tensor(state), step)
            next_state, reward, terminate,truncate,_ = env.step(action)
            done = (terminate or truncate)
            rewardEpi[i] += reward
            memory.add(action,state,reward,next_state,done)

            act, stat, rew, next_stat, do = memory.randomSample()
            lossEpi[i] += agent.update_parameter(act,stat,rew,next_stat,do)

            if(done):
                break
        
        state, _= env.reset()
        done = False

        if(((i % INTERVAL) == 0) and (i != 0)):
            print(f'\n {i}:reward max:{max(rewardEpi[i-INTERVAL:i])}, mean:{np.mean(rewardEpi[i-INTERVAL:i])}, min:{min(rewardEpi[i-INTERVAL:i])}')
            print(f' {i}:loss max:{max(lossEpi[i-INTERVAL:i])}, mean:{np.mean(lossEpi[i-INTERVAL:i])}, min:{min(lossEpi[i-INTERVAL:i])}\n')
    env.close()

    torch.save(agent.model,DQN_PARAM_PATH+str(NUM_EPI))

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