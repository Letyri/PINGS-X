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
    Data_res = args.data_res
    
    lr = args.lr
    epochs = args.epochs
    save_steps = args.save_step
    
    emb_size = args.emb_size
    depth = args.depth
    layer = args.layer
    pe = args.pe
    pe_L = args.pe_L
    
    if data_type == 'bur':
        in_l = 2
        out_l = 1
    elif data_type == 'bb':
        in_l = 2
        out_l = 2
    else:
        in_l = 2
        out_l = 3
    
    output_dir = args.output_dir\
        + '/XPINN/' + args.data_type + '_' + str(args.data_res) + '/' \
        + str(args.layer) + '/'
        
    if data_type == 'bur':
        
        _,_, _, xb, tb, ub, x_train, t_train, u_train, f_train = load_data_burgers_Grid_res(Data_res)
        
        Data = np.concatenate((x_train.reshape(-1,1),t_train.reshape(-1,1),u_train.reshape(-1,1)),axis=1)
        Data_b = np.concatenate((xb.reshape(-1,1),tb.reshape(-1,1),ub.reshape(-1,1)),axis=1)
        line_bank = np.array([1,0,0]).reshape(-1,3)
        d_inter = args.inter_d
        inter_bank = Block_control.Model_Decom_Overlap(Data,line_bank,2,d_inter)
        inter_bank_b = Block_control.Model_Decom_Overlap(Data_b,line_bank,2,d_inter)
        
        x1_train = torch.unsqueeze(torch.tensor(x_train[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        t1_train = torch.unsqueeze(torch.tensor(t_train[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        u1_train = torch.unsqueeze(torch.tensor(u_train[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zero1 = torch.unsqueeze(torch.tensor(f_train[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb1 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        tb1 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        ub1 = torch.unsqueeze(torch.tensor(ub[inter_bank_b[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x2_train = torch.unsqueeze(torch.tensor(x_train[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        t2_train = torch.unsqueeze(torch.tensor(t_train[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        u2_train = torch.unsqueeze(torch.tensor(u_train[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zero2 = torch.unsqueeze(torch.tensor(f_train[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb2 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        tb2 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        ub2 = torch.unsqueeze(torch.tensor(ub[inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x12_train = torch.unsqueeze(torch.tensor(x_train[inter_bank[:,0] * inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        t12_train = torch.unsqueeze(torch.tensor(t_train[inter_bank[:,0] * inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zero12 = torch.unsqueeze(torch.tensor(f_train[inter_bank[:,0] * inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb12 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,0] * inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        tb12 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,0] * inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zerob12 = torch.unsqueeze(torch.tensor(np.zeros(len(xb12)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        u1_gt = u1_train
        u2_gt = u2_train
        ub1_gt = ub1
        ub2_gt = ub2
        
        print("fluid1 data Num : ",len(x1_train))
        print("fluid2 data Num : ",len(x2_train))
        print("Boundary1 data Num : ",len(xb1))
        print("Boundary2 data Num : ",len(xb2))
        print("Interface data Num : ",len(x12_train))
        print("Boundary interface Num : ",len(xb12))
    
    elif data_type == 'bb':
        x, t, u, v, xb, tb, ub, vb, x_L, t_L = load_data_BB_Grid_res(Data_res)
        
        Data = np.concatenate((x.reshape(-1,1),t.reshape(-1,1)),axis=1)
        Data_b = np.concatenate((xb.reshape(-1,1),tb.reshape(-1,1)),axis=1)
        line_bank = np.array([1,0,0.00]).reshape(-1,3)
        d_inter = args.inter_d
        inter_bank = Block_control.Model_Decom_Overlap(Data,line_bank,2,d_inter)
        inter_bank_b = Block_control.Model_Decom_Overlap(Data_b,line_bank,2,d_inter)
        
        x1_train = torch.unsqueeze(torch.tensor(x[inter_bank[:,0]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        t1_train = torch.unsqueeze(torch.tensor(t[inter_bank[:,0]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        u1_train = torch.unsqueeze(torch.tensor(u[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        v1_train = torch.unsqueeze(torch.tensor(v[inter_bank[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zero1 = torch.unsqueeze(torch.tensor(np.zeros(len(x1_train)), dtype=torch.float32, requires_grad=True),-1).to(device)
        xb1 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,0]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb1 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,0]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub1 = torch.unsqueeze(torch.tensor(ub[inter_bank_b[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        vb1 = torch.unsqueeze(torch.tensor(vb[inter_bank_b[:,0]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x2_train = torch.unsqueeze(torch.tensor(x[inter_bank[:,1]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        t2_train = torch.unsqueeze(torch.tensor(t[inter_bank[:,1]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        u2_train = torch.unsqueeze(torch.tensor(u[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        v2_train = torch.unsqueeze(torch.tensor(v[inter_bank[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        zero2 = torch.unsqueeze(torch.tensor(np.zeros(len(x2_train)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb2 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,1]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb2 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,1]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub2 = torch.unsqueeze(torch.tensor(ub[inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        vb2 = torch.unsqueeze(torch.tensor(vb[inter_bank_b[:,1]==1], dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x12_train = torch.unsqueeze(torch.tensor(x[inter_bank[:,0] * inter_bank[:,1]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        t12_train = torch.unsqueeze(torch.tensor(t[inter_bank[:,0] * inter_bank[:,1]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero12 = torch.unsqueeze(torch.tensor(np.zeros(len(x12_train)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb12 = torch.unsqueeze(torch.tensor(xb[inter_bank_b[:,0] * inter_bank_b[:,1]==1]/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb12 = torch.unsqueeze(torch.tensor(tb[inter_bank_b[:,0] * inter_bank_b[:,1]==1]/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        zerob12 = torch.unsqueeze(torch.tensor(np.zeros(len(xb12)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        u1_gt = torch.concat((u1_train,v1_train),axis=1)
        u2_gt = torch.concat((u2_train,v2_train),axis=1)
        ub1_gt = torch.concat((ub1,vb1),axis=1)
        ub2_gt = torch.concat((ub2,vb2),axis=1)
        
        print("fluid1 data Num : ",len(x1_train))
        print("fluid2 data Num : ",len(x2_train))
        print("Boundary1 data Num : ",len(xb1))
        print("Boundary2 data Num : ",len(xb2))
        print("Interface data Num : ",len(x12_train))
        print("Boundary interface Num : ",len(xb12))
    
    model1 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    model2 = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
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
        pred_b1 = model1(xb1,tb1)
        pred_b2 = model2(xb2,tb2)
        
        if data_type=='bur':
            pred1,PDE_f1 = PDE_Equation.L_phy(x1_train,t1_train,model1,data_type)
            pred2,PDE_f2 = PDE_Equation.L_phy(x2_train,t2_train,model2,data_type)
            pred_12,_ = PDE_Equation.L_Burgers_XPINN(x12_train,t12_train,model1,model2)
            pred_b12,_ = PDE_Equation.L_Burgers_XPINN(xb12,tb12,model1,model2)
        
        elif data_type == 'bb':
            pred1,PDE_f1 = PDE_Equation.L_phy(x1_train,t1_train,model1,data_type,_x_L=x_L,_t_L=t_L)
            pred2,PDE_f2 = PDE_Equation.L_phy(x2_train,t2_train,model2,data_type,_x_L=x_L,_t_L=t_L)
            pred_12,_ = PDE_Equation.L_BB_XPINN(x12_train,t12_train,model1,model2,x_L,t_L)
            pred_b12,_ = PDE_Equation.L_BB_XPINN(xb12,tb12,model1,model2,x_L,t_L)
        
        PDE_Loss = criterion(PDE_f1,zero1) + criterion(PDE_f2,zero2)
        Data_Loss = criterion(pred1,u1_gt) + criterion(pred2,u2_gt)
        BD_Loss = criterion(pred_b1,ub1_gt) + criterion(pred_b2,ub2_gt)
        Interface_loss = criterion(pred_12,zero12) + criterion(pred_b12,zerob12) # + criterion(PDE_f_12,zero12)
        total_loss =  Data_Loss + PDE_Loss + BD_Loss + Interface_loss
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
    parser.add_argument('--output_dir', default='./results_avg', type=str, help='')
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data_type', default='bur', type=str, help='bur, bb, lid, L, Y')
    parser.add_argument('--data_res', default=100, type=int, help='100, 80, 50, 30')
    
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
    