import numpy as np
import torch
import torch.nn as nn
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

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(logits)
print(f"Predicted class: {y_pred}")
#tensor([[ 0.0820, -0.0153, -0.1121,  0.0854, -0.0653,  0.1364,  0.0372, -0.0080,
#         -0.0267,  0.0210]], device='cuda:0', grad_fn=<AddmmBackward0>)
#Predicted class: tensor([5], device='cuda:0')



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