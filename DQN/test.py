from collections import namedtuple
from collections import deque
import random
import numpy as np
import gymnasium as gym
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
'''

'''
loss_fn = F.mse_loss
agent = DQN(2,4,GAMMA,EPSILON)
Amax = torch.max(agent.model(torch.rand(1,4)))
print(f"Amax:{Amax}")
print(f"Amax.requires_grad:{Amax.requires_grad}")   #!!! これはTrueとなる!!!
y = 1.0 + 0.3 * Amax
print(f"y.requires_grad:{y.requires_grad}")
b = Amax.detach().clone()
print(f"b.requires_grad:{b.requires_grad}")
z = 1.0 + 0.3 * b
print(f"z.requires_grad:{z.requires_grad}")
print(f"agent.model(torch.rand(1,4))[0,0]:{agent.model(torch.rand(1,4))[0,0]}")
print(f"z:{z}")
w = loss_fn(agent.model(torch.rand(1,4))[0,0], z)
print(f"w.requires_grad:{w.requires_grad}")
#Amax:0.035308837890625
#Amax.requires_grad:True
#y.requires_grad:True
#b.requires_grad:False
#z.requires_grad:False
#agent.model(torch.rand(1,4))[0,0]:0.0145416259765625
#z:1.0107421875
#w.requires_grad:True

'''
'''
mse = lambda pred,targ:torch.pow((targ-pred),2)
a = torch.tensor((1.,2.,3.,4.),requires_grad=True)
b = torch.tensor((1.,4.,6.,8.),requires_grad=True)
m = mse(a,b).sum()              #(b-a)**2
print(f"m:{m}")
print(f"m.requires_grad:{m.requires_grad}")
m.backward()
print(a.grad)     #-2(b-a)    >>tensor([-0., -4., -6., -8.])
print(b.grad)     #2(b-a)*(b)     >>tensor([0., 4., 6., 8.])

c = torch.tensor((1.,2.,3.,4.),requires_grad=True)
d = torch.tensor((1.,4.,6.,8.),requires_grad=True)

nmse = nn.MSELoss(reduction='sum')
nm = nmse(c,d)
print(f"nm:{nm}")
print(f"nm.requires_grad.{nm.requires_grad}")
nm.backward()
print(f"c.grad:{c.grad}")
print(f"d.grad:{d.grad}")
'''

'''
a = np.array([1.,2.,3.,4.])
b = torch.as_tensor(a)    #as_tensor値を共有する！！
c = torch.tensor(a)       #tensrは新しくメモリを割り当てる
print(a)
print(b)
print(c)
c[2] = 0.
print(a)
print(b)
print(c)
b[2] = 0.
print(a)
print(b)
print(c)
'''
'''
relu = nn.ReLU()
print(relu(torch.rand(4)))
print(F.relu(torch.rand(4)))
'''
'''
model = torch.load('/home/yamamoto/DRL/DQN/param/DQNparam30000')
env = gym.make('CartPole-v1')
obs ,_ = env.reset()
print(obs)
print(model(torch.tensor(obs)))
'''

'''
hoge = namedtuple("test","a,b,c,d")
memory = deque([],maxlen=10)
_t = lambda x: torch.tensor([x])
memory.append(hoge(_t(1),_t(2),_t(3),_t(4)))
memory.append(hoge(_t(2),_t(4),_t(6),_t(8)))
memory.append(hoge(_t(3),_t(6),_t(9),_t(12)))
print(f"memory:{memory}")
fuga = random.sample(memory, 2)
print(f"fuga:{fuga}")
batch = hoge(*zip(*fuga))
print(f"batch:{batch}")
print(f"batch.b:{batch.b}")
print(f"torch.cat(batch.b):{torch.cat(batch.b)}")
'''
'''
a = torch.tensor(4).view(1,-1)
b = torch.tensor([2.,3.,4.,5.]).view(1,-1).dim()
print(a)
print(b)
'''

'''
agent = DQN(2,4,GAMMA,EPSILON)
state = torch.tensor(np.random.rand(10,4))
qList = agent.model(state)
rList = [100 for i in range(10)]
nList = torch.tensor([15 for i in range(15)])
dList = (i%3 == 0 for i in range(10))
actList = torch.tensor(np.random.randint(0,2,(10,1)),device=DEVICE)
print(actList)
print(qList)
choiceActList = torch.gather(qList,1,actList)
print(choiceActList)

y = torch.tensor(tuple(map(lambda r,n,d:
                    r + n
                    if not d
                    else 
                    r
                ,rList,nList,dList)), device=DEVICE)
print(y)
'''
qList = [[0,1] for _ in range(20)]
act = [[random.randint(0,1)] ]