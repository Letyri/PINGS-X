import numpy as np 
import torch
import os 
import wandb
import math 
import matplotlib.pyplot as plt 
from Data_generator import * 

from Model.model import PINN
from utils import utils,dataprocess,PDE_Equation,Block_control,interpolation
from configparser import ConfigParser

from utils.utils import dpdn
import time

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)

def main(args):
    # wandb.login()
    # wandb.init(project="ICLR_Baseline",
    #            name=f"Baseline_" + str(args.data_type) + '_XPINN_' + str(args.layer))
    
    args = args
    device = args.device
    data_type = args.data_type
    Data_num = args.data_num
    
    lr = args.lr
    epochs = args.epochs
    save_steps = args.save_step
    
    in_l = 2
    out_l = 3
    emb_size = args.emb_size
    depth = args.depth
    layer = args.layer
    pe = False
    pe_L = args.pe_L
    
    model1 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model2 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model3 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model4 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    data_dir = args.data_dir
    
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,Data_num)
    
    
    print("Decompose interface Area _ 1234")
    
    line1_L = np.array([-1,0,0.45,1]).reshape(-1,4)
    line1_R = np.array([-1,0,0.45,-1]).reshape(-1,4)
    line1 = np.array([-1,0,0.40,-1]).reshape(-1,4)
    r1 = math.sqrt(1**2 + 1**2)
    line2 = np.array([1,1,-0.8500-r1*0.05,-1]).reshape(-1,4)
    line2_L = np.array([1,1,-0.8500,-1]).reshape(-1,4)
    line2_R = np.array([1,1,-0.8500,1]).reshape(-1,4)
    r2 = math.sqrt(2**2 + (-1)**2)
    line3 = np.array([2,-1,-0.70-r2*0.05,-1]).reshape(-1,4)
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
    
    
    fluid1 = interpolation.block_decomposition_fluid(fluid_LR,line_bank1)
    fluid2 = interpolation.block_decomposition_fluid(fluid_LR,line_bank2)
    fluid3 = interpolation.block_decomposition_fluid(fluid_LR,line_bank3)
    fluid4 = interpolation.block_decomposition_fluid(fluid_LR,line_bank4)
    
    fluid12 = interpolation.block_decomposition_fluid(fluid_LR,line_bank12)
    fluid23 = interpolation.block_decomposition_fluid(fluid_LR,line_bank23)
    fluid24 = interpolation.block_decomposition_fluid(fluid_LR,line_bank24)
    
    
    print("Fluid 1 shape : ",len(fluid1))    
    print("Fluid 2 shape : ",len(fluid2))    
    print("Fluid 3 shape : ",len(fluid3))    
    print("Fluid 4 shape : ",len(fluid4))  
    print("Decomposition Boundary condition")
    line1 = np.array([-1,0,0.45,-1]).reshape(-1,4)
    line2 = np.array([1,1,-0.8500,-1]).reshape(-1,4)
    line3 = np.array([2,-1,-0.70,-1]).reshape(-1,4)
    
    wall1 , n1, wall2 , n2, wall3 , n3, wall4 , n4, \
    wall12, n12 , wb12_1,wb12_2, \
    wall23, n23 , wb23_2,wb23_3 , \
    wall24, n24 , wb24_2,wb24_4= \
        interpolation.block_interface_wall_Y(wall_LR,n,line1.squeeze(0),line2.squeeze(0),line3.squeeze(0),0.05)
        
    print("Wall 1 shape : ",len(wall1))    
    print("Wall 2 shape : ",len(wall2))    
    print("Wall 3 shape : ",len(wall3))    
    print("Wall 4 shape : ",len(wall4))    
    print("Fluid interface shape : ",len(wall12)+len(wall23)+len(wall24))    
    print("Fluid Total shape : ",len(wall1)+ len(wall2)+ len(wall3) + len(wall4))    

    print("Done")
    
    pt_x1, pt_y1, pt_wallx1, pt_wally1, u_ref1, v_ref1,u_wall1,v_wall1,n1, in_01, wall_01 = \
        dataprocess.variable_instance(fluid1,wall1,n1)
    pt_x2, pt_y2, pt_wallx2, pt_wally2, u_ref2, v_ref2,u_wall2,v_wall2,n2, in_02, wall_02 = \
        dataprocess.variable_instance(fluid2,wall2,n2)
    pt_x3, pt_y3, pt_wallx3, pt_wally3, u_ref3, v_ref3,u_wall3,v_wall3,n3, in_03, wall_03 = \
        dataprocess.variable_instance(fluid3,wall3,n3)
    pt_x4, pt_y4, pt_wallx4, pt_wally4, u_ref4, v_ref4,u_wall4,v_wall4,n4, in_04, wall_04 = \
        dataprocess.variable_instance(fluid4,wall4,n4)
    
    pt_x12, pt_y12, pt_wallx12, pt_wally12, u_ref12, v_ref12,u_wall12,v_wall12,n12, in_012, wall_012 = \
        dataprocess.variable_instance(fluid12,wall12,n12)
    pt_x23, pt_y23, pt_wallx23, pt_wally23, u_ref23, v_ref23,u_wall23,v_wall23,n23, in_023, wall_023 = \
        dataprocess.variable_instance(fluid23,wall23,n23)
    pt_x24, pt_y24, pt_wallx24, pt_wally24, u_ref24, v_ref24,u_wall24,v_wall24,n24, in_024, wall_024 = \
        dataprocess.variable_instance(fluid24,wall24,n24)
    
    zero_12 = torch.zeros((len(pt_x12),3)).to(device)
    zero_23 = torch.zeros((len(pt_x23),3)).to(device)
    zero_24 = torch.zeros((len(pt_x24),3)).to(device)
    
    ub1_gt = torch.concat((u_wall1,v_wall1),axis=1)
    ub2_gt = torch.concat((u_wall2,v_wall2),axis=1)
    ub3_gt = torch.concat((u_wall3,v_wall3),axis=1)
    ub4_gt = torch.concat((u_wall4,v_wall4),axis=1)
    u1_gt =  torch.concat((u_ref1,v_ref1),axis=1)
    u2_gt =  torch.concat((u_ref2,v_ref2),axis=1)
    u3_gt =  torch.concat((u_ref3,v_ref3),axis=1)
    u4_gt =  torch.concat((u_ref4,v_ref4),axis=1)
    
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam([
        {'params': model1.parameters()},
        {'params': model2.parameters()},
        {'params': model3.parameters()},
        {'params': model4.parameters()},
        ], lr = lr)
    
    output_dir = args.output_dir\
        + '/XPINN/' + args.data_type + '_' + str(args.data_num) + '/' \
        + str(args.layer) + '/'
    
    if not os.path.isdir(output_dir):
        print("***output directory make***")
        os.makedirs(output_dir)
    
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    
    for iters in range(1,epochs+1):
        optimizer.zero_grad()
        
        pred_b1 = model1(pt_wallx1,pt_wally1) 
        pred_b2 = model2(pt_wallx2,pt_wally2)
        pred_b3 = model3(pt_wallx3,pt_wally3)
        pred_b4 = model4(pt_wallx4,pt_wally4)
        
        BD_Loss = criterion(pred_b1[:,:2],ub1_gt) + criterion(pred_b2[:,:2],ub2_gt) +\
            criterion(pred_b3[:,:2],ub3_gt) + criterion(pred_b4[:,:2],ub4_gt)
        
        pred_1, PDE_f1 = PDE_Equation.L_phy(pt_x1, pt_y1, model1,data_type,_Re=Re)
        pred_2, PDE_f2 = PDE_Equation.L_phy(pt_x2, pt_y2, model2,data_type,_Re=Re)
        pred_3, PDE_f3 = PDE_Equation.L_phy(pt_x3, pt_y3, model3,data_type,_Re=Re)
        pred_4, PDE_f4 = PDE_Equation.L_phy(pt_x4, pt_y4, model4,data_type,_Re=Re)
        p1_x,p1_y = PDE_Equation.L_phy_b(pt_wallx1,pt_wally1,model1)
        p2_x,p2_y = PDE_Equation.L_phy_b(pt_wallx2,pt_wally2,model2)
        p3_x,p3_y = PDE_Equation.L_phy_b(pt_wallx3,pt_wally3,model3)
        p4_x,p4_y = PDE_Equation.L_phy_b(pt_wallx4,pt_wally4,model4)
        pred12 = PDE_Equation.L_NS_XPINN(pt_x12,pt_y12,model1,model2,Re)
        pred23 = PDE_Equation.L_NS_XPINN(pt_x23,pt_y23,model2,model3,Re)
        pred24 = PDE_Equation.L_NS_XPINN(pt_x24,pt_y24,model2,model4,Re)
        
        BD_Loss = BD_Loss + \
            criterion(dpdn(p1_x, p1_y, n1), wall_01) + criterion(dpdn(p2_x, p2_y, n2), wall_02) \
                + criterion(dpdn(p3_x, p3_y, n3), wall_03) + criterion(dpdn(p4_x, p4_y, n4), wall_04) 
        Data_Loss = criterion(pred_1[:,:2],u1_gt) + criterion(pred_2[:,:2],u2_gt) \
            + criterion(pred_3[:,:2],u3_gt) + criterion(pred_4[:,:2],u4_gt)
        PDE_Loss = criterion(PDE_f1,in_01) + criterion(PDE_f2,in_02) + criterion(PDE_f3,in_03) + criterion(PDE_f4,in_04)
        Interface_loss = criterion(pred12,zero_12) + criterion(pred23,zero_23) + criterion(pred24,zero_24)
        
        total_loss = Data_Loss + PDE_Loss + BD_Loss + Interface_loss
        total_loss.backward()
        optimizer.step()
                     
        # wandb.log({
        #         "Loss" : total_loss.item(),
        #         "PDEloss" : PDE_Loss.item()/2.0,
        #         "BDloss" : BD_Loss.item()/2.0,
        #         "Valid" : Data_Loss.item()/2.0,
        #         "Inter" : Interface_loss.item(),
        #             },step = iters)
    
        
        if iters % save_steps == 0:
            print("Saving model")
            path = os.path.join(output_dir, '{:02d}.tar'.format(iters))
            torch.save({
                "model_1" : model1.state_dict(),
                "model_2" : model2.state_dict(),
                "model_3" : model3.state_dict(),
                "model_4" : model4.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }, path)
            print("Saved checkpoint at ", path)
            
        if iters%10 ==0:
                    print("Iteration: ",iters, \
                        "Data_loss_u: ",Data_Loss.item())
    
                        
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./results', type=str, help='')
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data_type', default='bur', type=str, help='bur, bb, lid, L, Y')
    parser.add_argument('--data_num', default=10000, type=int, help='10000, 6400, 2500, 900')
    
    parser.add_argument('--epochs', default=100000, type=int, help='10000 / 100000')
    parser.add_argument('--lr', default=5e-3, type=float, help='5e-3 / 1e-3')
    parser.add_argument('--save_step', default=100, type=int, help='100 / 1000')
    
    parser.add_argument('--emb_size', default=128, type=int, help='128 / 256')
    parser.add_argument('--depth', default=5, type=int, help='5 / 8')
    parser.add_argument('--layer', default='Tanh', type=str, help='Tanh / Sin')
    parser.add_argument('--pe', default=False, type=bool, help='False / True')
    parser.add_argument('--pe_L', default=4, type=int, help='4 / 8')
    
    parser.add_argument('--inter_d', default=0.05, type=float, help='0.05 / 0.01')
    
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Excution time: {end_time - start_time:.2f} sec")
