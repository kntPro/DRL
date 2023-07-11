import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import *

class Model(nn.Module):
    def __init__(self, obsShape, actShape):   #obsShape:受け取る状態の次元, actShape:受け取る行動の次元
        super().__init__()
        self.first = nn.Linear(obsShape,16, dtype=torch.float32)
        self.firstAct = nn.ReLU() 
        self.second = nn.Linear(16,32, dtype=torch.float32)
        self.secondAct = nn.ReLU()
        self.third = nn.Linear(32,actShape, dtype=torch.float32)

    def forward(self,state):
        #x1 = torch.tensor(x,dtype=torch.float32,requires_grad=True).clone().detach().requires_grad_(True).cuda()
        #x = y
        t = self.first(state)
        t = self.firstAct(t)
        t = self.second(t)
        t = self.secondAct(t)
        t = self.third(t)

        return t 
    

class DQN():
    def __init__(self,actShape,obsShape,gamma,epsilon,test=False):
        self.model = Model(obsShape,actShape).to(DEVICE)
        #self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.gamma = gamma
        self.step = 0
        self.epsilon = lambda :max((EPSILON_COE-self.step)/EPSILON_COE, epsilon) #epsilon-annealing
        self.test = test

    def sample_action(self,x):
        if (self.epsilon() < np.random.rand()) or self.test:
            with torch.no_grad():
                logits = self.model(x)
                #probab = nn.Softmax(dim=1)(logits)
                predict = torch.argmax(logits).view(1,1).to(torch.int64)
        else:
            predict = torch.tensor([[np.random.randint(0,2)]],device=DEVICE,dtype=torch.int64,requires_grad=False)

        self.step+=1
        return predict
 
    def computeMaxQ(self,reward,nextState,done):
        retQList = []
        with torch.no_grad():
         for (r, n, d) in zip(reward, nextState, done):
            if not d:
                retQList.append(list(r + self.gamma * torch.max(self.model(n))))  
            else:
                retQList.append(list(r))
        
        return torch.tensor(retQList,device=DEVICE)

    def update_parameter(self, act, state, reward, next_state, done):
        '''
        y = torch.tensor(tuple(map(lambda r,n,d:
                    r + (self.gamma * torch.max(self.model(n)).detach().clone().cuda())
                    if not d
                    else 
                    r
                ,reward,next_state,done)), device=DEVICE).view(-1,1)
        '''        
        y = self.computeMaxQ(reward,next_state,done)
        qList = self.model(state)
        x = torch.gather(qList,1,act)

        loss = self.loss_fn(x,y)
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
        self.optimizer.step()
    
        return loss
    


def main():
    agent = DQN(2,4,GAMMA,EPSILON)
    rew = torch.tensor([3.], requires_grad=True)
    r = rew.detach().clone().float().cuda()
    y = r + (0.99 * torch.max(agent.model(torch.rand(4))).detach().clone().float().cuda())

    print(r)
    print(y)
    print(y.requires_grad)
    x = agent.model(torch.rand(4))[1]
    print(x)
    print(x.requires_grad)
    w = x.float()

if __name__ == '__main__':
    main()
