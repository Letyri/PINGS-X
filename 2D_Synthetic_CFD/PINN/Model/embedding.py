import torch
import numpy as np

from torch import nn
from math import *

class adaemb(nn.Module):
    def __init__(self,L):
        super().__init__()
        self.L = L
        self.PE = nn.Parameter(torch.ones(L))
    
    def forward(self,x):
        embed = []
        embed.append(x)
        for i in range(self.L):
            for fn in [torch.sin,torch.cos]:
                embed.append(fn(2. * pi * i * self.PE[i] *  x))
        
        
        return torch.cat(embed,-1)

class posemb(nn.Module):
    def __init__(self,L):
        super().__init__()
        self.L = L
    
    def forward(self,x):
        embed = []
        embed.append(x)
        for i in range(self.L):
            for fn in [torch.sin,torch.cos]:
                embed.append(fn(2. * pi * i * x))
        
        
        return torch.cat(embed,-1)