from Model.model import *
from configparser import ConfigParser

from Model.model import PINN
from utils import utils,dataprocess,interpolation,PDE_Equation
from configparser import ConfigParser

from utils.utils import dpdn

import matplotlib.pyplot as plt
import argparse

def main(args):
    device = 'cuda'
    in_l = 2
    out_l = 3
    emb_size = 128
    depth = 5
    layer = args.layer
    pe = False
    pe_L = 10
    
    data_dir = args.data_dir
    
    data_type = 'L'
    
    model1 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model2 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model3 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    load_path = args.load_path

    print("load weight from ",load_path)
    checkpoint = torch.load(load_path)
    model1.load_state_dict(checkpoint['model_1'])
    model2.load_state_dict(checkpoint['model_2'])
    model3.load_state_dict(checkpoint['model_3'])
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,100)
    
    print("Decompose interface Area _ 1234")
    print("Decompose interface Area _ 1234")
    
    line1_L = np.array([1,0,-0.3500,-1]).reshape(-1,4)
    line1 = np.array([1,0,-0.3500,1]).reshape(-1,4)
    
    line2 = np.array([0,1,-0.4500,-1]).reshape(-1,4)
    line2_U = np.array([0,1,-0.4500,1]).reshape(-1,4)
    
    line12_1 = np.array([1,0,-0.3000,1]).reshape(-1,4)
    line12_2 = np.array([1,0,-0.4000,-1]).reshape(-1,4)
    
    line23_2 = np.array([0,1,-0.4000,1]).reshape(-1,4)
    line23_3 = np.array([0,1,-0.5000,-1]).reshape(-1,4)


    line_bank1 = line1_L
    line_bank2 = np.concatenate((line1,line2),axis=0)
    line_bank3 = line2_U
    line_bank12 = np.concatenate((line12_1,line12_2),axis=0)
    line_bank23 = np.concatenate((line23_2,line23_3),axis=0)
    
    
    fluid1 = interpolation.block_decomposition_fluid(fluid,line_bank1)
    fluid2 = interpolation.block_decomposition_fluid(fluid,line_bank2)
    fluid3 = interpolation.block_decomposition_fluid(fluid,line_bank3)
    
    pt_x1, pt_y1, pt_wallx_1, pt_wally_1, u_ref_1, v_ref_1,u_wall_1,v_wall_1,n_1, in_0_1, wall_0_1 = \
        dataprocess.variable_instance(fluid1,wall,n)
    pt_x2, pt_y2, pt_wallx_2, pt_wally_2, u_ref_2, v_ref_2,u_wall_2,v_wall_2,n_2, in_0_2, wall_0_2 = \
        dataprocess.variable_instance(fluid2,wall,n)
    pt_x3, pt_y3, pt_wallx_3, pt_wally_3, u_ref_3, v_ref_3,u_wall_3,v_wall_3,n_3, in_0_3, wall_0_3 = \
        dataprocess.variable_instance(fluid3,wall,n)
    
    print("Fluid 1 shape : ",len(fluid1))    
    print("Fluid 2 shape : ",len(fluid2))    
    print("Fluid 3 shape : ",len(fluid3))       
    print("Fluid Total shape : ",len(fluid1)+ len(fluid2)+ len(fluid3))   
    
    pred_1, PDE_f1 = PDE_Equation.L_phy(pt_x1, pt_y1, model1,data_type,_Re=Re)
    pred_2, PDE_f2 = PDE_Equation.L_phy(pt_x2, pt_y2, model2,data_type,_Re=Re)
    pred_3, PDE_f3 = PDE_Equation.L_phy(pt_x3, pt_y3, model3,data_type,_Re=Re)
        
    F1 = torch.abs(PDE_f1).mean()
    F2 = torch.abs(PDE_f2).mean()
    F3 = torch.abs(PDE_f3).mean()

    print("PDE loss ",((F1+F2+F3)/3.0).item())
    
    Pred = torch.concat((pred_1,pred_2,pred_3),dim=0)
    u_ref = torch.concatenate((u_ref_1,u_ref_2,u_ref_3),dim=0)
    v_ref = torch.concatenate((v_ref_1,v_ref_2,v_ref_3),dim=0)
    u_gt = torch.concatenate((u_ref,v_ref),dim=1)
    pt_x = torch.concat((pt_x1,pt_x2,pt_x3),dim=0)
    pt_y = torch.concat((pt_y1,pt_y2,pt_y3),dim=0)
    print(u_gt.shape)
    Error = utils.relative_l2(Pred[:,:2],u_gt)
    print("Relative loss of v",Error.item())
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default=None, type=str, help='')
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--layer', default='Tanh', type=str)
    args = parser.parse_args()
    main(args)
    
