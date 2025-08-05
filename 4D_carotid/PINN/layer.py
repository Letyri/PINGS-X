import torch
import numpy as np

from torch import nn    


class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
                     
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class TanhLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh())
        
    def forward(self, x):
        out = self.NN(x)
        
        return out
    
class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.ReLU())
        
    def forward(self, x):
        out = self.NN(x)
        
        return out
