import copy
import numpy as np
import torch
import gymnasium as gym
from memory import *
from dqn import *
from tqdm import tqdm
from env import CartPoleFallReward, ClipedObsCart
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import os
from config import *


def main():
    env = CartPoleFallReward(gym.make('CartPole-v1'), fall=FALL_REWARD)
    #env = ClipedObsCart(gym.make('CartPole-v1'), fall=FALL_REWARD)
    state,_ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    memory = DequeMemory(MEMORY)    
    agent = DQN(2,env.observation_space.shape[0],GAMMA,EPSILON)
    rewardEpi = np.zeros(NUM_EPI)
    lossEpi =  np.zeros(NUM_EPI)
    done = False
    writer = SummaryWriter() 
    #train
    
    
    for i in tqdm(range(NUM_EPI)):
        for step in range(NUM_STEP):
            action = agent.sample_action(state)
            next_state, reward, terminate,truncate,_ = env.step(action.item())
            done = terminate or truncate
           # if(done): next_state = None
            rewardEpi[i] += reward

            next_state = torch.tensor(next_state,dtype=torch.float32,device=DEVICE).unsqueeze(0)
            reward = torch.tensor(reward,dtype=torch.float32,device=DEVICE).unsqueeze(0)

            memory.add(action,state,reward,next_state,done)
            state = next_state

            if(len(memory.memory)>=BATCH_SIZE):
                transitions = memory.randomSample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                batch_action = torch.cat(batch.action)
                batch_state = torch.cat(batch.state)
                batch_reward = torch.cat(batch.reward)
                batch_nextState = torch.cat(batch.next_state)
                batch_done = batch.done
                lossEpi[i] += agent.update_parameter(batch_action,batch_state,batch_reward,batch_nextState,batch_done)
            
            if(done):
                break
        
        state, _= env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        done = False

        if(((i % INTERVAL) == 0) and (i != 0)):
            intrvlMax = max(rewardEpi[i-INTERVAL:i]) 
            intrvlMin = min(rewardEpi[i-INTERVAL:i])
            intrvlMean = np.mean(rewardEpi[i-INTERVAL:i])
            lossMax = max(lossEpi[i-INTERVAL:i])
            lossMean = np.mean(lossEpi[i-INTERVAL:i])
            lossMin = min(lossEpi[i-INTERVAL:i])

            print(f'\n {i}:reward max:{intrvlMax}, mean:{intrvlMean}, min:{intrvlMin}')
            print(f' {i}:loss max:{lossMax}, mean:{lossMean}, min:{lossMin}\n')

            writer.add_scalar("RerardMax",intrvlMax,i)
            writer.add_scalar("RewardAve",intrvlMean,i)
            writer.add_scalar("RewardMin",intrvlMin,i)
            writer.add_scalar("LossMax",lossMax,i)
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
    #plt.show()
        
    

if __name__=="__main__":
    main()