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
        self.first = nn.Linear(obsShape,32)
        self.firstAct = nn.ReLU() 
        self.second = nn.Linear(32,32)
        self.secondAct = nn.ReLU()
        self.third = nn.Linear(32,actShape)

    @torch.autocast(device_type=DEVICE)
    def forward(self,x):
        #x1 = torch.tensor(x,dtype=torch.float32,requires_grad=True).clone().detach().requires_grad_(True).cuda()
        x1 = x.clone().detach().requires_grad_(True).cuda()
        x2 = self.first(x1)
        x3 = self.firstAct(x2)
        x4 = self.second(x3)
        x5 = self.secondAct(x4)
        x6 = self.third(x5)

        return x6
    

class DQN():
    def __init__(self,actShape,obsShape,gamma,epsilon,training=True):
        self.model = Model(obsShape,actShape).to(DEVICE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.gamma = gamma
        self.epsilon = lambda step: max(1e6-step/1e6, epsilon) 
        self.traing = training

    def sample_action(self,x, step):
        if (self.epsilon(step) < np.random.rand()) and self.traing:
            logits = self.model(x)
            probab = nn.Softmax(dim=0)(logits)
            tensorPredict = torch.argmax(probab).detach()
            predict = tensorPredict.cpu().numpy()
        else:
            predict = np.random.randint(0,2)

        return predict

    def update_parameter(self, act, obs, reward, next_obs, done):
        with torch.autocast(device_type=DEVICE, dtype=torch.float32):
            if done == True:
              y = reward
            else:
              y = reward + self.gamma * torch.max(self.model(next_obs))

            x = self.model(obs)[int(act.item())]
            loss = self.loss_fn(x,y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


