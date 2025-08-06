import numpy as np
import torch
import random
import math 
import time 

from torch.autograd import Variable
device = 'cuda'

def L_phy(x,y,model,data_type,_x_L:float=1.0,_t_L:float=10.0,_Re:float=0.3):
    sol = model(x, y)
    if data_type == 'bur':
        u = sol[:,0].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        f = Burgers(u,u_x,u_t,u_xx)
    
    elif data_type == 'bb':
        u = sol[:,0].reshape(-1,1)
        v = sol[:,1].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
        
        u_t = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        
        f1 = BB1(u,u_t/_t_L,u_x/_x_L,v_x/_x_L)
        f2 = BB2(u,v,v_t/_t_L,u_x/_x_L,v_x/_x_L,u_xxx/(_x_L)**3)
        f = f1 + f2
        
    else:
        u = sol[:,0].reshape(-1,1)
        v = sol[:,1].reshape(-1,1)
        p = sol[:,2].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
        
        Eu = u*u_x + v*u_y + p_x - (1/_Re)*(u_xx + u_yy)
        Ev = u*v_x + v*v_y + p_y - (1/_Re)*(v_xx + v_yy)
        Ec = u_x + v_y
        f = Eu + Ev + Ec
    
    return sol,f 

def L_phy_b(x,y,model):
    sol = model(x, y)
    p = sol[:,2].reshape(-1,1)
    p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    return p_x,p_y
           
def L_Burgers_XPINN(x,t,model1,model2):
    sol1 = model1(x, t)  # output 3 
    
    u1 = sol1[:,0].reshape(-1,1)
    u_x = torch.autograd.grad(u1.reshape(-1,1).sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u1.reshape(-1,1).sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    
    f1 = Burgers(u1,u_x,u_t,u_xx)
    
    sol2 = model2(x, t)  # output 3 
    u2 = sol2[:,0].reshape(-1,1)
    u_x = torch.autograd.grad(u2.reshape(-1,1).sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u2.reshape(-1,1).sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    
    f2 = Burgers(u2,u_x,u_t,u_xx)
    
    
    return (u1-u2),f1-f2 

def L_BB_XPINN(x,t,model1,model2,x_L,t_L):
    sol1 = model1(x, t)  # output 3 
    
    u1 = sol1[:,0].reshape(-1,1)
    v1 = sol1[:,1].reshape(-1,1)
    
    u_x = torch.autograd.grad(u1.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_x = torch.autograd.grad(v1.reshape(-1,1).sum(), x, create_graph=True)[0]
    
    u_t = torch.autograd.grad(u1.reshape(-1,1).sum(), t, create_graph=True)[0]
    v_t = torch.autograd.grad(v1.reshape(-1,1).sum(), t, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    
    f1_1 = BB1(u1,u_t/t_L,u_x/x_L,v_x/x_L)
    f1_2 = BB2(u1,v1,v_t/t_L,u_x/x_L,v_x/x_L,u_xxx/(x_L)**3)
    
    f1 = f1_1 + f1_2
    
    sol2 = model2(x, t)  # output 3 
    
    u2 = sol2[:,0].reshape(-1,1)
    v2 = sol2[:,1].reshape(-1,1)
    
    u_x = torch.autograd.grad(u2.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_x = torch.autograd.grad(v2.reshape(-1,1).sum(), x, create_graph=True)[0]
    
    u_t = torch.autograd.grad(u2.reshape(-1,1).sum(), t, create_graph=True)[0]
    v_t = torch.autograd.grad(v2.reshape(-1,1).sum(), t, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    
    f2_1 = BB1(u2,u_t/t_L,u_x/x_L,v_x/x_L)
    f2_2 = BB2(u2,v2,v_t/t_L,u_x/x_L,v_x/x_L,u_xxx/(x_L)**3)
    
    f2 = f2_1 + f2_2
    
    
    return sol1-sol2,f1-f2

def L_NS_XPINN(x,t,model1,model2,Re):
    sol1 = model1(x, t)  # output 3 
    sol2 = model2(x,t)
    
    return sol1 - sol2

def Burgers(u, u_x, u_y, u_xx): 
    return u_y + u * u_x - (0.01 / np.pi) * u_xx

def BB1(u, u_t, u_x, v_x): 
        return u_t - 2 * u * u_x - 0.5 * v_x

def BB2(u, v, v_t, u_x, v_x, u_xxx): 
    return v_t - 0.5 * u_xxx - 2 * u * v_x - 2 * u_x * v