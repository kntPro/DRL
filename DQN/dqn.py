import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Model(nn.Module):
    def __init__(self, obsShape, actShape):   #obsShape:受け取る状態の次元, actShape:受け取る行動の次元
        super.__init__()
        self.first = nn.Linear(obsShape,32)
        self.firstAct = nn.ReLU() 
        self.second = nn.Linear(32,32)
        self.secondAct = nn.ReLU()
        self.third = nn.Linear(32,actShape)

    def forward(self,x):
        x = torch.tensor(x)
        x = self.first(x)
        x = self.firstAct(x)
        x = self.second(x)
        x = self.secondAct(x)
        x = self.third(x)

        return x
    

class DQN():
    def __init__(self,obsShape,actShape,device):
        self.model = Model(obsShape,actShape).to(device)

    def sample_action(self,x):
        logits = self.model(x)
        probab = nn.Softmax(dim=1)(logits)
        predict = probab.argmax(1)

        return predict
