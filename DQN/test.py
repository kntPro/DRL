import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Opt
from dqn import *
from config import *
'''a = [[1.,2.,3.,4.],[5.,6.,7.,8.]]
na = np.array(a)
ta = torch.tensor(na)
print(a)
print(na)
print(ta)
'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = NeuralNetwork().to(device)

'''
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(logits)
print(f"Predicted class: {y_pred}")
#tensor([[ 0.0820, -0.0153, -0.1121,  0.0854, -0.0653,  0.1364,  0.0372, -0.0080,
#         -0.0267,  0.0210]], device='cuda:0', grad_fn=<AddmmBackward0>)
#Predicted class: tensor([5], device='cuda:0')
'''


class FuncNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.one=nn.Linear(28*28, 512)
        self.two=nn.ReLU()
        self.three=nn.Linear(512, 512)
        self.four=nn.ReLU()
        self.five=nn.Linear(512, 10)
        self.six=nn.Linear(512,5)

    def forward(self,x):
        x=self.flatten(x)
        x=self.one(x) 
        x=self.two(x)
        x=self.three(x)
        x=self.four(x)
        y=self.five(x)
        z=self.six(x)

        return y,z

'''
FModel = FuncNeuralNetwork().to(device)
print(FModel)
X = torch.rand(1,28,28,device=device)
logits1,logits2= FModel(X)
pred_probab1 = nn.Softmax(dim=1)(logits1)
pred_probab2 = nn.Softmax(dim=1)(logits2)
print(f"logits1: {logits1}")
print(f"probab1: {pred_probab1}")
print(f"logits2: {logits2}")
print(f"probab2: {pred_probab2}")
'''
'''
agent = DQN(2,4,GAMMA,EPSILON)
X = torch.rand(1,4,dtype=torch.float32)
print(X)
predict = agent.model(X)
print(f"predict:{predict}")
act = predict[0,1].requires_grad_(True)
a = torch.tensor(1.).cuda()
print(f'act:{act}')
print(f'type:{type(act)}')
print(f'a-predict:{a-predict}')
print(f'a-act:{a-act}')
print(agent.model(np.arange(4)))
'''
'''
agent = DQN(2,4,GAMMA,EPSILON)
opt = Opt.SGD(agent.model.parameters(), lr=1e-3)
tensor = torch.tensor([1.,2.,3.,4.])
print(agent.model(tensor)[0])
print(agent.model(tensor)[1])
loss = (agent.model(tensor)[0] - agent.model(tensor)[1]).pow(2)
lossF = F.mse_loss(input=agent.model(tensor)[1], target=agent.model(tensor)[0])
#print(loss)
#print(lossF)
loss.backward()
opt.step()
opt.zero_grad()
print(agent.model(tensor)[0])
print(agent.model(tensor)[1])
'''


agent = DQN(2,4,GAMMA,EPSILON)            #損失関数を二乗にした場合、２つのprint間で変化はなく、学習していないことがわかる
opt = Opt.SGD(agent.model.parameters(), lr=1e-3)
preTensor = torch.randn(4)
print(agent.model(preTensor)[0] - agent.model(preTensor)[1])
name = 0
if(name == 0):
    loss_fn = lambda x, y: (x-y).pow(2)
else:
    loss_fn = nn.MSELoss()

agent.model.train()
for i in tqdm(range(int(1e5))):
    tensor = torch.tensor(np.random.rand(4),dtype=torch.float32)
    loss = loss_fn(agent.model(tensor)[0], agent.model(tensor)[1])
    loss.backward()
    opt.step()
    opt.zero_grad()

agent.model.eval()
print(agent.model(preTensor)[0] - agent.model(preTensor)[1])



