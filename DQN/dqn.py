import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Model(nn.Module):
    def __init__(self, obsShape, actShape):   #obsShape:受け取る状態の次元, actShape:受け取る行動の次元
        super().__init__()
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
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3)

    def sample_action(self,x):
        logits = self.model(x)
        probab = nn.Softmax(dim=1)(logits)
        predict = probab.argmax(1)

        return predict



#以下pytorch勉強用
sample = Model(4,2).cuda()
opt = torch.optim.SGD(sample.parameters(),lr=1e-4)
opt.zero_grad()
x = torch.rand(5,4).cuda()
y = torch.rand(5,2).cuda()
z1 = sample(x)
loss = torch.pow(y-z1,2).sum()
loss.backward()
opt.step()
z2 = sample(x)
z3 = sample(x)

print(loss)
print(f"z1:{z1}")
print(f"z2:{z2}")
print(f"z3:{z3}")





