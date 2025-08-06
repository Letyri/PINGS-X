from Model.model import *
from configparser import ConfigParser

from Model.model import PINN
from utils import utils,dataprocess
from configparser import ConfigParser

from utils.utils import dpdn

import matplotlib.pyplot as plt
import argparse

def main():
    device='cuda'
    in_l = 2
    out_l = 3
    emb_size = 128
    depth = 5
    layer = args.layer
    pe = False
    pe_L = 10
    
    model = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    data_dir = args.data_dir
    load_path = args.load_path
    
    print("load weight from ",load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,100)
    print(len(fluid))
    
    pt_x, pt_y, pt_wallx, pt_wally, u_ref, v_ref,u_wall,v_wall,n, in_0, wall_0 = dataprocess.variable_instance(fluid,wall_LR,n)
    
    Pred = model(pt_x, pt_y)
    u = Pred[:,0].reshape(-1,1)
    v = Pred[:,1].reshape(-1,1)
    p = Pred[:,2].reshape(-1,1)
    
    u_x = torch.autograd.grad(u.reshape(-1,1).sum(), pt_x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.reshape(-1,1).sum(), pt_y, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v.reshape(-1,1).sum(), pt_x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.reshape(-1,1).sum(), pt_y, create_graph=True)[0]
    
    p_x = torch.autograd.grad(p.reshape(-1,1).sum(), pt_x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.reshape(-1,1).sum(), pt_y, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, pt_x, torch.ones(pt_x.shape).to(device), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, pt_y, torch.ones(pt_y.shape).to(device), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, pt_x, torch.ones(pt_x.shape).to(device), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, pt_y, torch.ones(pt_y.shape).to(device), create_graph=True)[0]
    
    Eu = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    Ev = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
    Ec = u_x + v_y
    
    PDE1 = (torch.abs(Eu).mean() + torch.abs(Ev).mean()+ torch.abs(Ec).mean())/3.0
    print(PDE1.item())
    
    u_gt = torch.concat((u_ref,v_ref),dim=1)
    print(u_gt.shape)
    Error = utils.relative_l2(Pred[:,:2],u_gt)
    print("Relative loss of v",Error.item())
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--layer', default='Tanh', type=str)
    args = parser.parse_args()
    main(args)
    
