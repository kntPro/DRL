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
        self.first = nn.Linear(obsShape,32, dtype=torch.float32)
        self.firstAct = nn.ReLU() 
        self.second = nn.Linear(32,32, dtype=torch.float32)
        self.secondAct = nn.ReLU()
        self.third = nn.Linear(32,actShape, dtype=torch.float32)

    @torch.autocast(device_type=DEVICE)
    def forward(self,x):
        #x1 = torch.tensor(x,dtype=torch.float32,requires_grad=True).clone().detach().requires_grad_(True).cuda()
        t = x.clone().detach().float().cuda()
        t = self.first(t)
        t = self.firstAct(t)
        t = self.second(t)
        t = self.secondAct(t)
        t = self.third(t)

        return t.float() 
    

class DQN():
    def __init__(self,actShape,obsShape,gamma,epsilon,test=False):
        self.model = Model(obsShape,actShape).to(DEVICE)
        self.loss_fn = nn.SmoothL1Loss()
        #self.loss_fn = nn.MSELoss()
        #self.loss_fn = lambda pred,targ:torch.pow((targ-pred),2)
        #self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.gamma = gamma
        self.step = 0
        self.epsilon = lambda :max((EPSILON_COE-self.step)/EPSILON_COE, epsilon) #epsilon-annealing
        self.test = test

    def sample_action(self,x):
        if (self.epsilon() < np.random.rand()) or self.test:
            with torch.no_grad():
                logits = self.model(x)
                probab = nn.Softmax(dim=0)(logits)
                tensorPredict = torch.argmax(probab).detach()
                predict = tensorPredict.cpu().numpy()
        else:
            predict = np.random.randint(0,2)

        self.step+=1

        return predict

    def update_parameter(self, act, obs, reward, next_obs, done):
        #reward = torch.clamp(rew, min=-1.0, max=1.0)
        with torch.autocast(device_type=DEVICE, dtype=torch.float32):
            if done == True:
                y = reward
            else:
                y = reward + (self.gamma * torch.max(self.model(next_obs)).detach().clone().cuda())

            x = self.model(obs)[int(act.item())]

        loss = self.loss_fn(x,y)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
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
