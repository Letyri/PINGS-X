import numpy as np
import torch
import random

from torch.autograd import Variable

device = 'cuda'

def relative_l2(u, u_gt):
    return torch.norm(u - u_gt) / torch.norm(u_gt)

def dpdn(p_x, p_y, n):
    p_n = p_x*n[:,0].reshape(-1,1) + p_y*n[:,1].reshape(-1,1)
    p_n.reshape(-1,1)
    return p_n

def MSELoss(pred,target):
    Loss = torch.sum((pred - target) ** 2, dim=1)
    
    return Loss 
    

