import numpy as np 
import torch
import scipy.io

from torch.autograd import Variable
torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)


device = 'cuda'

def preprocess_f(data_dir,N:int=6400,noise=False,noise_lv:float=0.01):
    
    data = scipy.io.loadmat(data_dir)
    
    for key, value in data.items():
        
        if key == 'fluid':
            fluid = value
        if key == 'wall':
            wall = value
        if key == 'n':
            n = value
        if key == 'fluidLR':
            fluid_LR = value
        if key == 'wallLR':
            wall_LR = value

    fluid = fluid[~np.isnan(fluid).any(axis=1)]
    if 'fluidLR' in data:
        fluid_LR = fluid_LR[~np.isnan(fluid_LR).any(axis=1)]
    # sampling x4
    # fluid_LR = fluid[::16].copy()
    L0 = np.min(np.concatenate((fluid[:,:2], wall[:,:2]), axis = 0))
    L1 = np.max(np.concatenate((fluid[:,:2], wall[:,:2]), axis = 0))
    if len(wall[0]) == 4:
        U = np.max(np.concatenate((fluid[:,2:], wall[:,2:]), axis = 0))
    else:
        U = np.max(np.concatenate((fluid[:,2:]), axis = 0))
    
    nu = 1e-4
    
    L = L1 - L0
    Re = U*(L1 - L0)/nu
    
    if 'fluidLR' not in data:
        fluid_LR = fluid[np.random.choice(fluid.shape[0], N, replace=False)]
    
    if noise:
        noise = noise_lv * np.mean(fluid[:,2:]) * np.random.rand(len(fluid_LR),2)
        fluid_LR[:,2:] = fluid_LR[:,2:] + noise 
    
    fluid[:,:2] = (fluid[:,:2] - L0) / (L1 - L0)
    fluid[:,2:] = (fluid[:,2:] )/ (U)
    wall[:,:2] = (wall[:,:2] - L0) / (L1 - L0)
    wall[:,2:] = (wall[:,2:] )/ (U)
    
    fluid_LR[:,:2] = (fluid_LR[:,:2] - L0) / (L1 - L0)
    fluid_LR[:,2:] = (fluid_LR[:,2:] )/ (U)
    
    if 'wallLR' in data:
        wall_LR[:,:2] = (wall_LR[:,:2] - L0) / (L1 - L0)
        wall_LR[:,2:] = (wall_LR[:,2:] )/ (U)
    else:
        wall_LR = wall
    
    return fluid,wall,n,fluid_LR,wall_LR,Re

def variable_instance(fluid,wall,n):
    pt_x = Variable(torch.from_numpy(fluid[:,0].reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(fluid[:,1].reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_wallx = Variable(torch.from_numpy(wall[:,0].reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_wally = Variable(torch.from_numpy(wall[:,1].reshape(-1,1)).float(), requires_grad=True).to(device)
    
    in0 = np.zeros((len(pt_x),1))
    wall0 = np.zeros((len(pt_wallx),1))
    
    in_0 = Variable(torch.from_numpy(in0).float(), requires_grad=False).to(device)
    wall_0 = Variable(torch.from_numpy(wall0).float(), requires_grad=False).to(device)
    
    u_ref = Variable(torch.from_numpy(fluid[:,2].reshape(-1,1)).float(), requires_grad=False).to(device)
    v_ref = Variable(torch.from_numpy(fluid[:,3].reshape(-1,1)).float(), requires_grad=False).to(device)
    
    if len(wall[0]) == 4:
        u_wall = Variable(torch.from_numpy(wall[:,2].reshape(-1,1)).float(), requires_grad=False).to(device)
        v_wall = Variable(torch.from_numpy(wall[:,3].reshape(-1,1)).float(), requires_grad=False).to(device)
    else:
        u_wall = Variable(torch.from_numpy(wall0).float(), requires_grad=False).to(device)
        v_wall = Variable(torch.from_numpy(wall0).float(), requires_grad=False).to(device)

    n = Variable(torch.from_numpy(n.reshape(-1,2)).float(), requires_grad=False).to(device)
    
    return pt_x, pt_y, pt_wallx, pt_wally, u_ref, v_ref,u_wall,v_wall,n, in_0, wall_0 
