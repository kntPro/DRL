import numpy as np
import torch
import torch.nn as nn
from dqn import *
import os

print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)
print('os.path.dirname(__file__):   ', os.path.dirname(__file__))
os.environ['PYTHONPATH'] = os.path.dirname(__file__)
print('getcwd:      ', os.getcwd())

'''a = [[1.,2.,3.,4.],[5.,6.,7.,8.]]
na = np.array(a)
ta = torch.tensor(na)
print(a)
print(na)
print(ta)
'''

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
'''
sample_model = Model(4,2)
torch.save(sample_model,'./param/sample')
copy_model = torch.load('./param/sample')
x = torch.rand(2,4).cuda()
print(f'sample:{sample_model(x)}')
'''