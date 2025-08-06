import numpy as np 
import torch
import os 
import wandb
import matplotlib.pyplot as plt 
from Data_generator import * 

from Model.model import PINN
from utils import utils,dataprocess,PDE_Equation
from configparser import ConfigParser

from utils.utils import dpdn

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)

def main(args):
    x, t, u, xb, tb, ub, x_train, t_train, u_train, f_train = load_data_burgers(100,False)
    device = 'cuda'
    
    x_test = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, requires_grad=True),-1).to(device)
    t_test = torch.unsqueeze(torch.tensor(t, dtype=torch.float32, requires_grad=True),-1).to(device)
    u_test = torch.unsqueeze(torch.tensor(u, dtype=torch.float32, requires_grad=True),-1).to(device)
    
    in_l = 21
    out_l = 1
    emb_size = 128
    depth = 5
    layer = args.layer
    # layer = 'Sin'
    pe = False
    pe_L = 10
    
    model = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    load_path = args.load_path
    print("load weight from ",load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    
    u_pred = model(x_test,t_test)
    pred = u_pred.data.cpu().numpy()
    print(u_pred.shape)
    print(u_test.reshape(-1,1).shape)
    Error = utils.relative_l2(u_pred,u_test.reshape(-1,1))
    print("Relative loss of v",Error.item())
    
      
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='Tanh', type=str)
    parser.add_argument('--load_path', default=None, type=str)
    args = parser.parse_args()
    main(args)