import numpy as np 
import torch
import os 
import wandb
from Data_generator import * 

from Model.model import PINN
from utils import utils,dataprocess,PDE_Equation,Block_control
from configparser import ConfigParser
from utils.utils import dpdn

import matplotlib.pyplot as plt
import argparse

def main(args):
    x, t, u, xb, tb, ub, x_train, t_train, u_train, f_train = load_data_burgers(10000)
    device = 'cuda'
    
    u = u.reshape(-1)
    Data = np.concatenate((x.reshape(-1,1),t.reshape(-1,1)),axis=1)
    line_bank = np.array([1,0,0.00]).reshape(-1,3)
    d_inter = 0.00
    inter_bank = Block_control.Model_Decom_Overlap(Data,line_bank,2,d_inter)
    
    x1_test = torch.unsqueeze(torch.tensor(x[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    t1_test = torch.unsqueeze(torch.tensor(t[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    u1_test = torch.unsqueeze(torch.tensor(u[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    
    x2_test = torch.unsqueeze(torch.tensor(x[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    t2_test = torch.unsqueeze(torch.tensor(t[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    u2_test = torch.unsqueeze(torch.tensor(u[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
    
    
    in_l = 2
    out_l = 1
    emb_size = 128
    depth = 5
    layer = args.layer
    pe = False
    pe_L = 4
    
    Model1 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    Model2 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    load_path = args.load_path
    print("load weight from ",load_path)
    checkpoint = torch.load(load_path)
    Model1.load_state_dict(checkpoint['model1'])
    Model2.load_state_dict(checkpoint['model2'])
    
    Pred1 = Model1(x1_test, t1_test)
    Pred2 = Model2(x2_test, t2_test)
    u_x1 = torch.autograd.grad(Pred1.sum(), x1_test, create_graph=True)[0]
    u_t1 = torch.autograd.grad(Pred1.sum(), t1_test, create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1, x1_test, torch.ones(x1_test.shape).to(device), create_graph=True)[0]
    
    f1 = PDE_Equation.Burgers(Pred1,u_x1,u_t1,u_xx1)
    
    u_x2 = torch.autograd.grad(Pred2.sum(), x2_test, create_graph=True)[0]
    u_t2 = torch.autograd.grad(Pred2.sum(), t2_test, create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2, x2_test, torch.ones(x2_test.shape).to(device), create_graph=True)[0]
    
    f2 = PDE_Equation.Burgers(Pred2,u_x2,u_t2,u_xx2)
    
    
    PDE = (torch.abs(f1).mean() + torch.abs(f2).mean())/2.0
    print(PDE.item())
    
    Pred = torch.concat((Pred1,Pred2),dim=0)
    posx =  torch.concat((x1_test,x2_test),dim=0)
    post =  torch.concat((t1_test,t2_test),dim=0)
    gt_u = torch.concat((u1_test,u2_test),dim=0)
    print(Pred.shape)
    print(gt_u.shape)
    Error = utils.relative_l2(Pred,gt_u)
    print("Relative loss of v",Error.item())
    print("")
    
    plt.scatter(posx.cpu().detach(),post.cpu().detach(), c=Pred.data.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()  
    
    
    plt.scatter(posx.cpu().detach(),post.cpu().detach(), c=torch.abs(gt_u.reshape(-1)-Pred.reshape(-1)).data.cpu().numpy()\
        , cmap='viridis', vmax=0.01)
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()  
    
    
    print("MSE loss of u_total",
          (np.sqrt(((gt_u.reshape(-1)).data.cpu().numpy() - Pred.reshape(-1).data.cpu().numpy())**2).mean()))
    print("")
    
     
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default=None, type=str, help='')
    parser.add_argument('--layer', default='Tanh', type=str)
    args = parser.parse_args()
    main(args)
    
