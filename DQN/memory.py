import numpy as np
import torch

class Memory():
    def __init__(self, size, actShape, obsShape) :
        self.size = size
        self.action = torch.tensor(()).new_zeros((size,actShape)).cuda()
        self.obs = torch.tensor(()).new_zeros((size,obsShape)).cuda()
        self.reward = torch.tensor(()).new_zeros((size,1)).cuda()
        self.next_obs = torch.tensor(()).new_zeros((size,obsShape)).cuda()
        self.done = torch.tensor(()).new_zeros((size,1)).cuda()
        self.index = 0

    def add(self, action, obs, reward, next_obs, done):
        try:
            self.action[self.index] = torch.tensor(action)
            self.obs[self.index] = torch.tensor(obs)
            self.reward[self.index] = torch.tensor(reward)
            self.next_obs[self.index] = torch.tensor(next_obs) 
            self.done[self.index] = torch.tensor(done) 
        except(IndexError) as e:
            print(e)
            print('no more added')

        self.index += 1

    def randomSample(self):
        index = np.random.randint(0,self.index)

        try:
            action = self.action[index]
            obs = self.obs[index]
            reward = self.reward[index]
            next_obs = self.next_obs[index]
            done = self.done[index]
        except(IndexError) as e:
            print(e)
            print(f'{index} is no assign')

        return action, obs, reward, next_obs, done
    
    def reset(self):
        self.action.zero_()
        self.obs.zero_()
        self.reward.zero_()
        self.next_obs.zero_()
        self.done.zero_()
        self.index  = 0

class NumMemory:
    def __init__(self, size, actShape, obsShape) :
        self.size = size
        self.action = np.zeros((self.size, actShape)) 
        self.obs = np.zeros((self.size, obsShape))
        self.reward = np.zeros((self.size, 1)) 
        self.next_obs = np.zeros((self.size, obsShape))
        self.done = np.zeros((self.size, 1))
        self.index = 0

    def add(self, action, obs, reward, next_obs, done):
        try:
            self.action[self.index] = action
            self.obs[self.index] = obs
            self.reward[self.index] = reward
            self.next_obs[self.index] = next_obs
            self.done[self.index] = done 
        except(IndexError) as e:
            print(e)
            print('no more added')

        self.index += 1

    def randomSample(self):
        index = np.random.randint(0,self.index)

        try:
            action = self.action[index]
            obs = self.obs[index]
            reward = self.reward[index]
            next_obs = self.next_obs[index]
            done = self.done[index]
        except(IndexError) as e:
            print(e)
            print(f'{index} is no assign')

        return action, obs, reward, next_obs, done
    
    def reset(self):
        np.zeros_like(self.action)
        np.zeros_like(self.obs) 
        np.zeros_like(self.reward) 
        np.zeros_like(self.next_obs) 
        np.zeros_like(self.done) 
        self.index  = 0
