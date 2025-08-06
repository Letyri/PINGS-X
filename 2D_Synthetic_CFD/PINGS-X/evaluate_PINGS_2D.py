import numpy as np 
import torch
import os 
import wandb
import matplotlib.pyplot as plt 
from Data_generator import * 

#from Model.Gaussian_splat_Model import Gaussian_PINN
from Model.Gaussian_splat_Model_mean import Gaussian_PINN
from utils import utils,dataprocess_Navier,PDE_Equation,Gaussian
from configparser import ConfigParser


torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)

def main(args):
    device = "cuda"
    
    data_dir = args.data_dir

    Scale_factor = 100.0
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess_Navier.preprocess_f(data_dir)
    
    # x = fluid[:,0]
    # y = fluid[:,1]
    # u = fluid[:,2]
    # v = fluid[:,3]

    # GT point - train point
    LR_point = [(temp[0], temp[1]) for temp in fluid_LR]
    fluid_test = []
    for temp in fluid:
        if (temp[0], temp[1]) not in LR_point:
            fluid_test.append(temp)
    fluid_test = np.array(fluid_test)
    x = fluid_test[:,0]
    y = fluid_test[:,1]
    u = fluid_test[:,2]
    v = fluid_test[:,3]
     
    x_test = torch.unsqueeze(torch.tensor(Scale_factor * x, dtype=torch.float32, requires_grad=True),-1).to(device)
    y_test = torch.unsqueeze(torch.tensor(Scale_factor * y, dtype=torch.float32, requires_grad=True),-1).to(device)
    u_test = torch.unsqueeze(torch.tensor(u, dtype=torch.float32, requires_grad=True),-1).to(device)
    v_test = torch.unsqueeze(torch.tensor(v, dtype=torch.float32, requires_grad=True),-1).to(device)
    
    pos_test = torch.concat((x_test,y_test),axis=1)
    
    load_path = args.load_path
    
    checkpoint = torch.load(load_path)
    print(checkpoint.keys())
    mean = (checkpoint['Mean'])
    Gaussian_scale = (checkpoint['Scale'])
    Gaussian_rotation = (checkpoint['Rot'])
    U_scale = (checkpoint['Property'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--Z_thred', default=0.0001, type=float, help='0.01 / 0.001 / 0.0001 / 0.00001')
    args = parser.parse_args()
    Model = Gaussian_PINN(args,mean,Gaussian_scale,Gaussian_rotation,U_scale).to(device)
    print(len(mean))
    
    Pred = Model(pos_test)
    Eu,Ev,Ec = PDE_Equation.L_NS_Gaussian_PDE(pos_test,Model,Scale_factor,Re)
    PDE_error = torch.abs(Eu+Ev+Ec).mean()
    u_gt = torch.concat((u_test,v_test),dim=1)
    Error = utils.relative_l2(Pred[:,:2],u_gt)
    print(f"Relative loss of {100 * Error.item():.2f}")
      
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str)
    args = parser.parse_args()
    main(args)