from Model.model import *
from configparser import ConfigParser

from Model.model import PINN
from utils import utils,dataprocess,Block_control,PDE_Equation
from configparser import ConfigParser

import matplotlib.pyplot as plt
import argparse

def main(args):
    device='cuda'
    in_l = 2
    out_l = 3
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
    
    data_dir = args.data_dir
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,100)
    
    line_bank = np.array([1,0,-0.50]).reshape(-1,3)
    d_inter = 0.0
    Num = 2
    
    fluid_bank = Block_control.Model_Decom_Overlap(fluid,line_bank,Num,d_inter)
    wall_bank = Block_control.Model_Decom_Overlap(wall_LR,line_bank,Num,d_inter)
     
    pt_x1, pt_y1, pt_wallx1, pt_wally1, u_ref1, v_ref1,u_wall1,v_wall1,n1, in_01, wall_01 = \
    dataprocess.variable_instance(fluid[fluid_bank[:,0]==1],wall_LR[wall_bank[:,0]==1],n[wall_bank[:,0]==1])
    pt_x2, pt_y2, pt_wallx2, pt_wally2, u_ref2, v_ref2,u_wall2,v_wall2,n2, in_02, wall_02 = \
    dataprocess.variable_instance(fluid[fluid_bank[:,1]==1],wall_LR[wall_bank[:,1]==1],n[wall_bank[:,1]==1])
    
    pred_1, PDE_f1 = PDE_Equation.L_phy(pt_x1, pt_y1, Model1,'lid',_Re=Re)
    pred_2, PDE_f2 = PDE_Equation.L_phy(pt_x2, pt_y2, Model2,'lid',_Re=Re)
        
    
    print("PDE loss ",torch.abs(PDE_f1+PDE_f2).mean().item())
    
    Pred = torch.concat((pred_1,pred_2),dim=0)
    u_ref = torch.concatenate((u_ref1,u_ref2),dim=0)
    v_ref = torch.concatenate((v_ref1,v_ref2),dim=0)
    u_gt = torch.concatenate((u_ref,v_ref),dim=1)
    pt_x = torch.concat((pt_x1,pt_x2),dim=0)
    pt_y = torch.concat((pt_y1,pt_y2),dim=0)
    
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
    
