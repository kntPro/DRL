import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class Model(nn.Module):
    def __init__(self,shape):   #shape:受け取る状態の次元
        super.__init__()
        self.first = nn.Linear(shape,32)
        self.second = 
