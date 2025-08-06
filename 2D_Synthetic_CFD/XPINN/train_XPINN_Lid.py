import numpy as np 
import torch
import os 
import wandb
import matplotlib.pyplot as plt 
from Data_generator import * 

from Model.model import PINN
from utils import utils,dataprocess,PDE_Equation,Block_control
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
    Data_res = args.data_res # don't touch
    
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
    
    data_dir = args.data_dir
        
    fluid,wall,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir)
    
    line_bank = np.array([1,0,-0.50]).reshape(-1,3)
    
    d_inter = args.inter_d
    Num = 2
    
    inter_bank = Block_control.Model_Decom_Overlap(fluid_LR,line_bank,Num,d_inter)
    inter_bank_w = Block_control.Model_Decom_Overlap(wall_LR,line_bank,Num,d_inter)
    
    fluid_bank = Block_control.Model_Decom_Overlap(fluid_LR,line_bank,Num,d_inter)
    wall_bank = Block_control.Model_Decom_Overlap(wall_LR,line_bank,Num,d_inter)
    
    pt_x1, pt_y1, pt_wallx1, pt_wally1, u_ref1, v_ref1,u_wall1,v_wall1,n1, in_01, wall_01 = \
    dataprocess.variable_instance(fluid_LR[fluid_bank[:,0]==1],wall_LR[wall_bank[:,0]==1],n[wall_bank[:,0]==1])
    pt_x2, pt_y2, pt_wallx2, pt_wally2, u_ref2, v_ref2,u_wall2,v_wall2,n2, in_02, wall_02 = \
    dataprocess.variable_instance(fluid_LR[fluid_bank[:,1]==1],wall_LR[wall_bank[:,1]==1],n[wall_bank[:,1]==1])
    pt_x12, pt_y12, pt_wallx12, pt_wally12, u_ref12, v_ref12,u_wall12,v_wall12,n12, in_012, wall_012 = \
    dataprocess.variable_instance(fluid_LR[inter_bank[:,0]*inter_bank[:,1]==1],wall_LR[inter_bank_w[:,0]*inter_bank_w[:,1]==1],n[inter_bank_w[:,0]*inter_bank_w[:,1]==1])
    
    zero_12 = torch.zeros((len(pt_x12),3)).to(device)
    
    ub1_gt = torch.concat((u_wall1,v_wall1),axis=1)
    ub2_gt = torch.concat((u_wall2,v_wall2),axis=1)
    u1_gt =  torch.concat((u_ref1,v_ref1),axis=1)
    u2_gt =  torch.concat((u_ref2,v_ref2),axis=1)
    
    
    print("fluid1 data Num : ",len(pt_x1))
    print("fluid2 data Num : ",len(pt_x2))
    print("Boundary1 data Num : ",len(pt_wallx1))
    print("Boundary2 data Num : ",len(pt_wallx2))
    print("Interface data Num : ",len(pt_x12))
    print("Boundary interface Num : ",len(pt_wallx12))
    
    model1 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model2 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    output_dir = args.output_dir\
        + '/XPINN/' + args.data_type + '_' + str(args.data_res) + '/' \
        + str(args.layer) + '/'
    
    
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam([
        {'params': model1.parameters()},
        {'params': model2.parameters()},
        ], lr = lr)
    
    if not os.path.isdir(output_dir):
        print("***output directory make***")
        os.makedirs(output_dir)
    
    
    for iters in range(1,epochs+1):
        model1.train()
        model2.train()
        optimizer.zero_grad()
        
        pred_b1 = model1(pt_wallx1,pt_wally1) 
        pred_b2 = model2(pt_wallx2,pt_wally2)
        
        BD_Loss = criterion(pred_b1[:,:2],ub1_gt) + criterion(pred_b2[:,:2],ub2_gt)
        
        pred_1, PDE_f1 = PDE_Equation.L_phy(pt_x1, pt_y1, model1,data_type,_Re=Re)
        pred_2, PDE_f2 = PDE_Equation.L_phy(pt_x2, pt_y2, model2,data_type,_Re=Re)
        p1_x,p1_y = PDE_Equation.L_phy_b(pt_wallx1,pt_wally1,model1)
        p2_x,p2_y = PDE_Equation.L_phy_b(pt_wallx2,pt_wally2,model2)
        pred12 = PDE_Equation.L_NS_XPINN(pt_x12,pt_y12,model1,model2,Re)
        
        BD_Loss = BD_Loss  + criterion(dpdn(p1_x, p1_y, n1), wall_01) + criterion(dpdn(p2_x, p2_y, n2), wall_02) 
        Data_Loss = criterion(pred_1[:,:2],u1_gt) + criterion(pred_2[:,:2],u2_gt)
        PDE_Loss = criterion(PDE_f1,in_01) + criterion(PDE_f2,in_02)
        Interface_loss = criterion(pred12,zero_12)
        
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
                "model1" : model1.state_dict(),
                "model2" : model2.state_dict(),
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
    parser.add_argument('--data_res', default=100, type=int, help='100, 80, 50, 25')
    
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
