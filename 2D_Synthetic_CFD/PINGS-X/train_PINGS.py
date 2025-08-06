import numpy as np 
import torch
import os 
import wandb
import matplotlib.pyplot as plt 
from Data_generator import * 
import argparse
import alphashape

from Model.Gaussian_splat_Model_mean import Gaussian_PINN
from utils import Gaussian,dataprocess_Navier,PDE_Equation,utils
from configparser import ConfigParser
from utils import Gaussian
from torch.autograd import Variable
import time
from tqdm import tqdm

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)

from scipy.spatial import cKDTree
def _grid_center_impl(pts, k, inside_ratio=0.5):
    (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
    dx, dy = (xmax - xmin) / k, (ymax - ymin) / k
    cx = xmin + dx * (0.5 + np.arange(k))
    cy = ymin + dy * (0.5 + np.arange(k))
    centers = np.stack(np.meshgrid(cx, cy), -1).reshape(-1, 2)   # (k*k, 2)

    tree = cKDTree(pts)
    dist, nearest = tree.query(centers)     # (k*k,)

    threshold = inside_ratio * min(dx, dy)
    keep_mask = dist < threshold
    kept_idx = np.unique(nearest[keep_mask])

    return kept_idx

def main(args):
    args = args
    device = args.device
    data_type = args.data_type
    Data_res = args.data_res # don't touch
    Data_num = args.data_num
    Gaussian_res = args.initial_point # initial point
    
    lr = args.lr
    epochs = args.epochs
    save_steps = args.save_step
    
    thr_dense = args.dense_thred
    thr_split = args.split_thred
    density_steps = args.dense_step
    Merge_step = args.merge_step 

    output_dir = args.output_dir_base\
        +args.data_type + '_' + str(args.data_res) + '_' + str(args.lr) +\
        '/Split_'+str(args.dense_step)+\
        '/Merge_'+str(args.merge_step)+\
        '/PDE/'\
        +str(Gaussian_res)+'/'
        

    alpha_shape = None

    
    if data_type == 'bur':
        Scale_factor = 100.0
        
        _, _, _, _, _, _, x_train, t_train, u_train, f_train = load_data_burgers_Grid_res(Gaussian_res)
        x_mean = torch.unsqueeze(torch.tensor(Scale_factor * x_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_mean = torch.unsqueeze(torch.tensor(Scale_factor * t_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_init = torch.unsqueeze(torch.tensor(u_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        mean = torch.concat((x_mean,t_mean),axis=1).detach().requires_grad_(True)
        dist = torch.clamp_min(Gaussian.find_closest_distances_PDE(mean), 0.01)
        eps = torch.log(dist)[...,None].repeat(1, 2)
        Gaussian_scale = (eps.detach()).requires_grad_()
        Gaussian_rotation = torch.zeros((len(mean)),device=device).requires_grad_()
        prop_init = (u_init.detach()).requires_grad_()
        U_scale = (prop_init.detach()).requires_grad_()
        print("# Initial Gaussian points :", mean.shape)
        
        _, _, _, xb, tb, ub, x_train, t_train, u_train, f_train = load_data_burgers_Grid_res(Data_res)
        x_train = torch.unsqueeze(torch.tensor(Scale_factor * x_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(Scale_factor * t_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(f_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb = torch.unsqueeze(torch.tensor(Scale_factor * xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb = torch.unsqueeze(torch.tensor(Scale_factor * tb, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        pos = torch.concat((x_train,t_train),axis=1)
        pos_b = torch.concat((xb,tb),axis=1)
        pos_total = torch.concat((pos,pos_b),axis=0)
        zero_total = torch.concat((zero_f,zero_b),axis=0)
        U_total = torch.concat((u_train,ub),axis=0)
        U_b = ub
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos))
        print("Total data Num : ",len(pos_total))
    
    elif data_type == 'bb':
        Scale_factor1 = 1.0 * 4.0
        Scale_factor2 = 5.0 * 4.0
        x, t, u, v, _, _, _, _, _,_ = load_data_BB_Grid_res(Gaussian_res)
        x_train = torch.unsqueeze(torch.tensor(Scale_factor1 * x, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(Scale_factor2 * t, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u, dtype=torch.float32, requires_grad=True),-1).to(device)
        v_train = torch.unsqueeze(torch.tensor(v, dtype=torch.float32, requires_grad=True),-1).to(device)
        
        mean = torch.concat((x_train,t_train),axis=1).clone().detach().requires_grad_(True)
        dist = torch.clamp_min(Gaussian.find_closest_distances_PDE(mean), 0.001)
        eps = torch.log(dist)[...,None].repeat(1, 2)
        prop_init = torch.concat((u_train,v_train),axis=1)
        Gaussian_scale = (eps.detach()).requires_grad_()
        Gaussian_rotation = torch.zeros((len(mean)),device=device).requires_grad_()
        U_scale = (prop_init.detach()).requires_grad_()
        print("# Initial Gaussian points :", mean.shape)
        
        x, t, u, v, xb, tb, ub, vb, _,_ = load_data_BB_Grid_res(Data_res)
        x_train = torch.unsqueeze(torch.tensor(Scale_factor1 * x, dtype=torch.float32, requires_grad=True),-1).to(device)
        t_train = torch.unsqueeze(torch.tensor(Scale_factor2 * t, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u, dtype=torch.float32, requires_grad=True),-1).to(device)
        v_train = torch.unsqueeze(torch.tensor(v, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(np.zeros(len(x)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        xb = torch.unsqueeze(torch.tensor(Scale_factor1 * xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        tb = torch.unsqueeze(torch.tensor(Scale_factor2 * tb, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        pos = torch.concat((x_train,t_train),axis=1)
        pos_b = torch.concat((xb,tb),axis=1)
        pos_total = torch.concat((pos,pos_b),axis=0)
        zero_total = torch.concat((zero_f,zero_b),axis=0)
        u_total = torch.concat((u_train,ub),axis=0)
        v_total = torch.concat((v_train,vb),axis=0)
        U_total = torch.concat((u_total,v_total),axis=1)
        U_b = torch.concat((ub,vb),axis=1)
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos))
        print("Total data Num : ",len(pos_total))
    
    else:
        Scale_factor = 100.0
        data_dir = args.data_dir
        
        _,_,n,fluid_LR,wall_LR,Re = dataprocess_Navier.preprocess_f(data_dir,Data_num)
        
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
            
        xb = torch.unsqueeze(torch.tensor(Scale_factor * xb, dtype=torch.float32, requires_grad=True),-1).to(device)
        yb = torch.unsqueeze(torch.tensor(Scale_factor * yb, dtype=torch.float32, requires_grad=True),-1).to(device)
        ub = torch.unsqueeze(torch.tensor(ub, dtype=torch.float32, requires_grad=True),-1).to(device)
        vb = torch.unsqueeze(torch.tensor(vb, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(xb)), dtype=torch.float32, requires_grad=True),-1).to(device)
        
        x_train = torch.unsqueeze(torch.tensor(Scale_factor * x_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        y_train = torch.unsqueeze(torch.tensor(Scale_factor * y_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        u_train = torch.unsqueeze(torch.tensor(u_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        v_train = torch.unsqueeze(torch.tensor(v_train, dtype=torch.float32, requires_grad=True),-1).to(device)
        zero_f = torch.unsqueeze(torch.tensor(np.zeros(len(x_train)), dtype=torch.float32, requires_grad=True),-1).to(device)

        n = Variable(torch.from_numpy(n.reshape(-1,2)).float(), requires_grad=False).to(device)
        
        pos = torch.concat((x_train,y_train),axis=1)
        pos_b = torch.concat((xb,yb),axis=1)
        pos_total = torch.concat((pos,pos_b),axis=0)
        
        ##
        grow_factor = 1.3
        pts = pos_total.detach().cpu().numpy()
        pts = np.round(pts, decimals=8)

        # initial k : sqrt(Gaussian_res)
        k = int(np.ceil(np.sqrt(Gaussian_res) * 1.1))

        for _ in range(10):
            idx_unique = _grid_center_impl(pts, k, inside_ratio=0.5)
            if idx_unique.size >= Gaussian_res:
                break
            k = int(k * grow_factor) + 1      
        else:
            raise RuntimeError("max_iter 안에 Gaussian_res 만큼 확보 못 함.")
        
        # Gaussian_res 개만 사용
        idx = rng1.choice(idx_unique, Gaussian_res, replace=False)

        prop = torch.concat((u_train,v_train,zero_f),axis=1)
        prop_b = torch.concat((ub,vb,zero_b),axis=1)
        prop_total = torch.concat((prop,prop_b),axis=0)

        mean_init = pos_total[idx]
        prop_init = prop_total[idx]
        mean = mean_init.clone().detach().requires_grad_(True)
        dist = torch.clamp_min(Gaussian.find_closest_distances_PDE(mean), 0.00001)
        eps = torch.log(dist)[...,None].repeat(1, 2)
        Gaussian_scale = (eps.detach()).requires_grad_()
        Gaussian_rotation = torch.zeros((len(mean)),device=device).requires_grad_()
        U_scale = (prop_init.detach()).requires_grad_()
        print("# Initial Gaussian points :", mean.shape)
        
        zero_total = torch.concat((zero_f,zero_b),axis=0)
        u_total = torch.concat((u_train,ub),axis=0)
        v_total = torch.concat((v_train,vb),axis=0)
        U_total = torch.concat((u_total,v_total),axis=1)
        U_b = torch.concat((ub,vb),axis=1)
        print("fluid data Num : ",len(pos))
        print("Boundary data Num : ",len(pos_b))
        print("Total data Num : ",len(pos_total))
        points = pos_b.cpu().detach().numpy()
        alpha_shape = alphashape.alphashape(points,alpha=.1)
      
    Model = Gaussian_PINN(
        args, mean,Gaussian_scale,Gaussian_rotation,U_scale,lr, thr_dense,thr_split,alpha_shape).to(device)
    
    criterion = torch.nn.MSELoss().to(device)
    Model_dir = os.path.join(output_dir,'Weights')
    Mean_dir = os.path.join(output_dir,'Mean')
    Scale_dir = os.path.join(output_dir,'Scale')
    
    if not os.path.isdir(output_dir):
        print("***output directory make***")
        os.makedirs(Model_dir)
        os.makedirs(Mean_dir)
        os.makedirs(Scale_dir)
    
    
    plt.scatter(Model.mean[:,0].cpu().detach(),Model.mean[:,1].cpu().detach())
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    fig_path = os.path.join(Mean_dir, 'Mean00.png')
    plt.savefig(fig_path)
    plt.close()
    
    Scale = Model.scale
    Plot_Scale = (Scale[:,0] + Scale[:,1]) / 2.0
    plt.scatter(Model.mean[:,0].cpu().detach(),Model.mean[:,1].cpu().detach(), c=Plot_Scale.data.cpu().numpy(), cmap='viridis')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar()
    fig_path = os.path.join(Scale_dir, '0.png')
    plt.savefig(fig_path)
    plt.close()
    

    print("Saving fig")
    Scale = Model.scale
    Plot_Scale = (Scale[:,0] + Scale[:,1]) / 2.0
    plt.scatter(Model.mean[:,0].cpu().detach(),Model.mean[:,1].cpu().detach(), c=Plot_Scale.data.cpu().numpy(), cmap='viridis')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar()
    fig_path = os.path.join(Scale_dir, '{:02d}.png'.format(0))
    plt.savefig(fig_path)
    plt.close()
    
    print("Saving model")
    path = os.path.join(Model_dir, '{:02d}.tar'.format(0))
    torch.save({
        "Mean" : Model.mean,
        "Rot" : Model.rot,
        "Scale" : Model.scale,
        "Property" : Model.property,
    }, path)
    print("Saved checkpoint at ", path)
            
    for iters in tqdm(range(1,epochs+1)):
        Model.optimizer.zero_grad()
        pred = Model(pos_total)
        if data_type == 'bb':
            se_u = utils.MSELoss(pred[:,0:1],U_total[:, 0:1])
            se_v = utils.MSELoss(pred[:,1:2],U_total[:, 1:2])
            pt_Data_Loss = se_u + se_v
            Data_Loss = se_u.mean() + se_v.mean()
        else:
            pt_Data_Loss = utils.MSELoss(pred[:,:2],U_total)
            Data_Loss = pt_Data_Loss.mean()

        if data_type == 'bur':
            PDE_f  = PDE_Equation.L_bur_Gaussian_PDE(pos_total,Model,Scale_factor)
            PDE_Loss = criterion(PDE_f,zero_total)
        elif data_type == 'bb':
            f1, f2 = PDE_Equation.L_bb_Gaussian_PDE_3(pos_total, Model, Scale_factor1, Scale_factor2)
            PDE_Loss = criterion(f1,zero_total) + criterion(f2,zero_total)
        else:
            PDE_u,PDE_v,PDE_c = PDE_Equation.L_NS_Gaussian_PDE(pos_total,Model,Scale_factor,Re)
            PDE_Loss = (criterion(PDE_u,zero_total) + criterion(PDE_v,zero_total) + criterion(PDE_c,zero_total))
        
        total_loss =  Data_Loss + PDE_Loss
        
        total_loss.backward()
        if iters%density_steps == 0 and iters > 100 :
            L_grad = Model.mean.grad

            Z = Model.PDF(pos_total)
            Z_sum = torch.sum(Z,dim=1).unsqueeze(-1)
            W = Z / (Z_sum + 1e-10)
            
            if data_type == 'bur':
                pt_PDE_Loss = PDE_f.squeeze()**2
            elif data_type == 'bb':
                pt_PDE_Loss = f1.squeeze()**2 + f2.squeeze()**2
            else:
                pt_PDE_Loss = PDE_u.squeeze()**2 + PDE_v.squeeze()**2 + PDE_c.squeeze()**2
            pt_total_loss = pt_Data_Loss + pt_PDE_Loss

            err = pt_total_loss.unsqueeze(-1)

            num = (W * err).sum(0)
            Gaussian_loss = num / (W.sum(0) + 1e-10)
    
        
        Model.optimizer.step()
        
        wandb.log({
                "Loss" : total_loss.item(),
                "PDE_loss" : PDE_Loss.item(),
                # "BDloss" : BD_Loss.item(),
                "Data_loss" : Data_Loss.item(),
                    },step = iters)
        
        if iters % save_steps == 0:
            
            print("Saving fig")
            Scale = Model.scale
            Plot_Scale = (Scale[:,0] + Scale[:,1]) / 2.0
            plt.scatter(Model.mean[:,0].cpu().detach(),Model.mean[:,1].cpu().detach(), c=Plot_Scale.data.cpu().numpy(), cmap='viridis')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.colorbar()
            fig_path = os.path.join(Scale_dir, '{:02d}.png'.format(iters))
            plt.savefig(fig_path)
            plt.close()
            
            print("Saving model")
            path = os.path.join(Model_dir, '{:02d}.tar'.format(iters))
            torch.save({
                "Mean" : Model.mean,
                "Rot" : Model.rot,
                "Scale" : Model.scale,
                "Property" : Model.property,
            }, path)
            print("Saved checkpoint at ", path)
        
        if iters%density_steps == 0 and iters > 100 :
            fig_path = os.path.join(Mean_dir, 'Mean{:02d}.png'.format(iters))
            Model.splating(L_grad, Gaussian_loss, fig_path)

        if iters % Merge_step == 0:
            Model.Gaussian_merge(pos_total)      
            
        if iters%10 ==0:
            print("Iteration: ",iters, \
                        "Data_loss_u: ",Data_Loss.item(),"# Mean : ", len(Model.mean))
        
                  
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_base', default='./results_GS_mean', type=str, help='Base directory for saving results.')
    parser.add_argument('--data_dir', default=None, type=str)
    
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data_type', default='bur', type=str, help='bur, bb, lid, L, Y')
    parser.add_argument('--data_res', default=80, type=int, help='100, 80, 50, 30')
    parser.add_argument('--data_num', default=6400, type=int, help='10000, 6400, 2500, 900')
    parser.add_argument('--initial_point', default=20, type=int, help='10 / 20 / 30 / 40')
    parser.add_argument('--Z_thred', default=0.0001, type=float, help='0.01 / 0.001 / 0.0001 / 0.00001')
    
    parser.add_argument('--epochs', default=10000, type=int, help='10000 / 100000')
    parser.add_argument('--lr', default=0.01, type=float, help='5e-3 / 1e-3')
    parser.add_argument('--save_step', default=100, type=int, help='100 / 1000')
    
    parser.add_argument('--dense_thred', default=0.0002, type=float, help='0.002 / 0.0002')
    parser.add_argument('--split_thred', default=2.0, type=float, help='2.0 / 1.0 / 0.0 / -1.0')
    parser.add_argument('--dense_step', default=100, type=int, help='100 / 500 / 10000')
    parser.add_argument('--merge_step', default=100, type=int, help='100 / 500 / 10000')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Excution time: {end_time - start_time:.2f} sec")
