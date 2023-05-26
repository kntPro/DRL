import numpy as np
import torch
import gym
from memory import *
from dqn import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    writer = SummaryWriter() 
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
            intrvlMax = max(rewardEpi[i-INTERVAL:]) 
            intrvlMin = min(rewardEpi[i-INTERVAL:])
            intrvlMean = np.mean(rewardEpi[i-INTERVAL:])
            lossMax = max(lossEpi[i-INTERVAL:])
            lossMean = np.mean(lossEpi[i-INTERVAL:])
            lossMin = min(lossEpi[i-INTERVAL:])

            print(f'\n {i}:reward max:{intrvlMax}, mean:{intrvlMean}, min:{intrvlMin}')
            print(f' {i}:loss max:{lossMax}, mean:{lossMean}, min:{lossMin}\n')

            writer.add_scalar("RewardAve",intrvlMax,i)
            writer.add_scalar("RerardMax",intrvlMax,i)
            writer.add_scalar("RewardMin",intrvlMin,i)
            writer.add_scalar("LossMaX",lossMax,i)
            writer.add_scalar("LossMean",lossMean,i)
            writer.add_scalar("LossMin",lossMin,i)
        
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