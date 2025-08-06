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
import time

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)

def main(args):
    # wandb.login()
    # wandb.init(project="ICLR_Time",
    #            name=f"Baseline_" + str(args.data_type) + '_' + str(args.layer) + '_')
    
    args = args
    device = args.device
    data_type = args.data_type
    
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
        + '/PINN/' + args.data_type + '/' \
        + str(args.layer) + '/'
        
    if data_type == 'bur':
        
        x, t, _, xb, tb, ub, x_train, t_train, u_train, f_train = load_data_burgers_Grid_res(100)
        x_train = torch.unsqueeze(torch.tensor(x_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(t_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(f_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb = torch.unsqueeze(torch.tensor(tb, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        pos = torch.concat((x_train,t_train),axis=1)
        pos_b = torch.concat((xb,tb),axis=1)
        
        u_gt = u_train
        ub_gt = ub
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos_b))
    
    elif data_type == 'bb':
        x, t, u, v, xb, tb, ub, vb, x_L, t_L = load_data_BB_Grid_res(100)
        x_train = torch.unsqueeze(torch.tensor(x/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(t/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u, dtype=torch.float32, requires_grad=True),-1).to(device)
        v_train = torch.unsqueeze(torch.tensor(v, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(np.zeros(len(x)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb = torch.unsqueeze(torch.tensor(xb/x_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb = torch.unsqueeze(torch.tensor(tb/t_L, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        pos = torch.concat((x_train,t_train),axis=1)
        pos_b = torch.concat((xb,tb),axis=1)
        u_gt = torch.concat((u_train,v_train),axis=1)
        ub_gt = torch.concat((ub,vb),axis=1)
        
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos_b))
        
    else:
        data_dir = args.data_dir
        
        _,_,n,fluid_LR,wall_LR,Re = dataprocess.preprocess_f(data_dir,6400)
        
        x_train = fluid_LR[:,0]
        y_train = fluid_LR[:,1]
        u_train = fluid_LR[:,2]
        v_train = fluid_LR[:,3]
        xb = wall_LR[:,0]
        yb = wall_LR[:,1]
        
        if data_type == 'lid':
            ub = wall_LR[:,2]
            vb = wall_LR[:,3]
        else:
            ub = np.zeros_like(xb)
            vb = np.zeros_like(xb)
            
        xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb = torch.unsqueeze(torch.tensor(yb, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x_train = torch.unsqueeze(torch.tensor(x_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(y_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        v_train = torch.unsqueeze(torch.tensor(v_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(np.zeros(len(x_train)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        n = torch.from_numpy(n.reshape(-1,2)).float().to(device)
    
        pos = torch.concat((x_train,t_train),axis=1)
        pos_b = torch.concat((xb,tb),axis=1)
        
        u_gt = torch.concat((u_train,v_train),axis=1)
        ub_gt = torch.concat((ub,vb),axis=1)
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos))
        
    model = PINN(in_l,out_l,emb_size,depth,layer,10,pe,pe_L).to(device)
    
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    if not os.path.isdir(output_dir):
        print("***output directory make***")
        os.makedirs(output_dir)
    
    for iters in range(1,epochs+1):
        model.train()
        optimizer.zero_grad()
        pred_b = model(xb,tb)
        BD_Loss = criterion(pred_b[:,:2],ub_gt)
        if data_type=='bur':
            pred,PDE_f = PDE_Equation.L_phy(x_train,t_train,model,data_type)
        elif data_type == 'bb':
            breakpoint()
            pred,PDE_f = PDE_Equation.L_phy(x_train,t_train,model,data_type,_x_L=x_L,_t_L=t_L)
        else:
            pred,PDE_f = PDE_Equation.L_phy(x_train,t_train,model,data_type,_Re=Re)
            p_x,p_y = PDE_Equation.L_phy_b(xb,tb,model)
            BD_Loss = BD_Loss + criterion(dpdn(p_x, p_y, n), zero_b) 
        
        PDE_Loss = criterion(PDE_f,zero_f)
        Data_Loss = criterion(pred[:,:2],u_gt)
        total_loss =  Data_Loss + PDE_Loss + BD_Loss 
        total_loss.backward()
        optimizer.step()
                 
        # wandb.log({
        #         "Loss" : total_loss.item(),
        #         "PDEloss" : PDE_Loss.item(),
        #         "BDloss" : BD_Loss.item(),
        #         "Valid" : Data_Loss.item(),
        #             },step = iters)
    
        
        if iters % save_steps == 0:
            print("Saving model")
            path = os.path.join(output_dir, '{:02d}.tar'.format(iters))
            torch.save({
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }, path)
            print("Saved checkpoint at ", path)
            
        if iters%10 ==0:
                    print("Iteration: ",iters, \
                        "Data_loss_u: ",Data_Loss.item())
    
      
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./results', type=str, help='')
    parser.add_argument('--data_dir', default=None, type=str, help='')
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data_type', default='bur', type=str, help='bur, bb, lid, L, Y')
    
    parser.add_argument('--epochs', default=100000, type=int, help='10000 / 100000')
    parser.add_argument('--lr', default=5e-3, type=float, help='5e-3 / 1e-3')
    parser.add_argument('--save_step', default=100, type=int, help='100 / 1000')
    
    parser.add_argument('--emb_size', default=128, type=int, help='128 / 256')
    parser.add_argument('--depth', default=5, type=int, help='5 / 8')
    parser.add_argument('--layer', default='Tanh', type=str, help='Tanh / Sin')
    parser.add_argument('--pe', default=False, type=bool, help='False / True')
    parser.add_argument('--pe_L', default=10, type=int, help='4 / 8')
    
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Excution time: {end_time - start_time:.2f} sec")
