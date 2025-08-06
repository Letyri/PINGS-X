from Model.model import *
from configparser import ConfigParser

from Model.model import PINN
from utils import utils,dataprocess,Block_control,interpolation,PDE_Equation
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
    model4 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    load_path = args.load_path
    
    print("load weight from ",load_path)
    checkpoint = torch.load(load_path)
    model1.load_state_dict(checkpoint['model_1'])
    model2.load_state_dict(checkpoint['model_2'])
    model3.load_state_dict(checkpoint['model_3'])
    model4.load_state_dict(checkpoint['model_4'])
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,100)
    print("Decompose interface Area _ 1234")
    
    line1_L = np.array([-1,0,0.45,1]).reshape(-1,4)
    line1_R = np.array([-1,0,0.45,-1]).reshape(-1,4)
    line1 = np.array([-1,0,0.40,-1]).reshape(-1,4)
    
    line2 = np.array([1,1,-0.8500,-1]).reshape(-1,4)
    line2_L = np.array([1,1,-0.8500,-1]).reshape(-1,4)
    line2_R = np.array([1,1,-0.8500,1]).reshape(-1,4)
    
    line3 = np.array([2,-1,-0.70,-1]).reshape(-1,4)
    line3_L = np.array([2,-1,-0.70,-1]).reshape(-1,4)
    line3_R = np.array([2,-1,-0.70,1]).reshape(-1,4)
    line_U = np.array([0,1,-0.3300,1]).reshape(-1,4)
    line_D = np.array([0,1,-0.3300,-1]).reshape(-1,4)
    
    line_bank1 = line1_L
    line_bank2 = np.concatenate((line1_R,line2_L,line3_L),axis=0)
    line_bank3 = np.concatenate((line2_R,line_U),axis=0)
    line_bank4 = np.concatenate((line3_R,line_D),axis=0)
    
    line_bank12 = np.concatenate((line1,line1_L),axis=0)
    line_bank23 = np.concatenate((line2,line2_R,line_U),axis=0)
    line_bank24 = np.concatenate((line3,line3_R,line_D),axis=0)
    
    
    fluid1 = interpolation.block_decomposition_fluid(fluid,line_bank1)
    fluid2 = interpolation.block_decomposition_fluid(fluid,line_bank2)
    fluid3 = interpolation.block_decomposition_fluid(fluid,line_bank3)
    fluid4 = interpolation.block_decomposition_fluid(fluid,line_bank4)
     
    pt_x_1, pt_y_1, pt_wallx_1, pt_wally_1, u_ref_1, v_ref_1,u_wall_1,v_wall_1,n_1, in_0_1, wall_0_1 = \
        dataprocess.variable_instance(fluid1,wall,n)
    pt_x_2, pt_y_2, pt_wallx_2, pt_wally_2, u_ref_2, v_ref_2,u_wall_2,v_wall_2,n_2, in_0_2, wall_0_2 = \
        dataprocess.variable_instance(fluid2,wall,n)
    pt_x_3, pt_y_3, pt_wallx_3, pt_wally_3, u_ref_3, v_ref_3,u_wall_3,v_wall_3,n_3, in_0_3, wall_0_3 = \
        dataprocess.variable_instance(fluid3,wall,n)
    pt_x_4, pt_y_4, pt_wallx_4, pt_wally_4, u_ref_4, v_ref_4,u_wall_4,v_wall_4,n_4, in_0_4, wall_0_4 = \
        dataprocess.variable_instance(fluid4,wall,n)
        
    Pred_1, PDE_f1 = PDE_Equation.L_phy(pt_x_1, pt_y_1, model1,data_type,_Re=Re)
    Pred_2, PDE_f2 = PDE_Equation.L_phy(pt_x_2, pt_y_2, model2,data_type,_Re=Re)
    Pred_3, PDE_f3 = PDE_Equation.L_phy(pt_x_3, pt_y_3, model3,data_type,_Re=Re)
    Pred_4, PDE_f4 = PDE_Equation.L_phy(pt_x_4, pt_y_4, model4,data_type,_Re=Re)
    F1 = torch.abs(PDE_f1).mean()
    F2 = torch.abs(PDE_f2).mean()
    F3 = torch.abs(PDE_f3).mean()
    F4 = torch.abs(PDE_f4).mean()

    print("PDE loss ",(F1+F2+F3+F4).item())
    
    Pred = torch.concat((Pred_1,Pred_2,Pred_3,Pred_4),dim=0)
    u_ref = torch.concatenate((u_ref_1,u_ref_2,u_ref_3,u_ref_4),dim=0)
    v_ref = torch.concatenate((v_ref_1,v_ref_2,v_ref_3,v_ref_4),dim=0)
    u_gt = torch.concatenate((u_ref,v_ref),dim=1)
    pt_x = torch.concat((pt_x_1,pt_x_2,pt_x_3,pt_x_4),dim=0)
    pt_y = torch.concat((pt_y_1,pt_y_2,pt_y_3,pt_y_4),dim=0)
    
    print(u_gt.shape)
    Error = utils.relative_l2(Pred[:,:2],u_gt)
    print("Relative loss of v",Error.item())
    print("")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default=None, type=str, help='')
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--layer', default='Tanh', type=str)
    args = parser.parse_args()
    main(args)
    
