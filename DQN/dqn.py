import torch
import numpy as np
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

    @torch.autocast(device_type='cuda')
    def forward(self,x):
        x = torch.tensor(x,dtype=torch.float32).cuda()
        x = self.first(x)
        x = self.firstAct(x)
        x = self.second(x)
        x = self.secondAct(x)
        x = self.third(x)

        return x
    

class DQN():
    def __init__(self,actShape,obsShape,gamma,epsilon):
        self.model = Model(obsShape,actShape).to('cuda')
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.gamma = gamma
        self.epsilon = epsilon

    def sample_action(self,x):
        if self.epsilon < np.random.rand():
            logits = self.model(x)
            probab = nn.Softmax(dim=0)(logits)
            tensorPredict = torch.argmax(probab).detach()
            predict = tensorPredict.cpu().numpy()
        else:
            predict = np.random.randint(0,2)

        return predict

    def update_parameter(self, obs, reward, next_obs, done):
        if done == True:
            y = reward
        else:
            y = reward + self.gamma * torch.max(self.model(next_obs))
        
        loss = torch.sum(torch.pow(y-self.model(obs),2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



'''
#以下pytorch勉強用
sample = Model(4,2).cuda()
opt = torch.optim.SGD(sample.parameters(),lr=1e-4)
opt.zero_grad()
x1 = torch.rand(1,4).cuda()
x2 = torch.rand(4).cuda()
x = torch.rand(5,4).cuda()
y = torch.rand(5,2).cuda()
z1 = sample(x)
x1out = sample(x1)
x2out = sample(x2)
loss = torch.pow(y-z1,2).sum()
loss.backward()
opt.step()
z2 = sample(x)
z3 = sample(x)

print(f'y:{y}')
print(f'x1out:{x1out}')
print(f'x2out:{x2out}')
'''
'''
sample = Model(4,2).cuda()
x = np.random.rand(1,4)
z = torch.rand(1,4).cuda()
y_z = sample(z)
y = sample(x)
print(y.shape)              #torch.Size([1, 2])
print(torch.max(y))         #tensor(-0.0412, device='cuda:0', grad_fn=<MaxBackward1>
print(torch.max(y).shape)   #torch.Size([])
'''

'''
print(loss)
print(f"z1:{z1}")
print(f"z2:{z2}")
print(f"z3:{z3}")
'''
'''
zero = 0
count = int(1e6)
for i in range(count):
    if np.random.randint(0,2) == 0:
        zero += 1
    
print(f'0_count:{zero}')
print(f'1_count{count-zero}')
'''