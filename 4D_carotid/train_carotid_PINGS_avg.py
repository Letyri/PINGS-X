import numpy as np 
import torch
import os 
import matplotlib.pyplot as plt 
import argparse

from utils import dataprocess_MRI

from Model.PINGS import Gaussian_PINN
# from Model.PINGS_wo_norm import Gaussian_PINN
# from Model.PINGS_wo_mean import Gaussian_PINN
from torch.autograd import Variable
import time
from tqdm import tqdm

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)
rng1 = np.random.RandomState(777)


def main(args):
    initial_gaussian_res = args.initial_gaussian_res
    args = args
    device = args.device
    
    lr = args.lr
    Z_thred = args.Z_thred
    epochs = args.epochs
    batch_size = args.batch_size
    alpha_shape = None

    thr_dense = args.dense_thred
    thr_split = args.split_thred

    save_steps = args.save_step
    density_steps = args.dense_step
    Merge_step = args.merge_step 

    output_dir = f"{args.output_dir}_{args.res}_{initial_gaussian_res}_{density_steps}_{Merge_step}"

    # wandb.login()
    # wandb.init(project="ICCV_carotid",
    #             name=f"carotid_PINGS-X",)

    device = args.device
    t_mf = args.t_mf

    data_dir =  args.data_dir
    Scale_factor = args.scale_factor

    res = args.res
    coord,vel,L0,L1,L_gt,U_gt = dataprocess_MRI.preprocess_carotid_avg_2(data_dir,t_mf,res)
    T_scale = U_gt / L_gt
    t = np.ones((coord.shape[0], 1)) * (t_mf * 0.25 * T_scale)

    coord_0,vel_0,_,_,_,_ = dataprocess_MRI.preprocess_carotid_avg_2(data_dir,t_mf-2,res)
    t_0 = np.ones((coord_0.shape[0], 1)) * ((t_mf-2) * 0.25 * T_scale)
    coord_1,vel_1,_,_,_,_ = dataprocess_MRI.preprocess_carotid_avg_2(data_dir,t_mf-1,res)
    t_1 = np.ones((coord_1.shape[0], 1)) * ((t_mf-1) * 0.25 * T_scale)
    coord_2,vel_2,_,_,_,_ = dataprocess_MRI.preprocess_carotid_avg_2(data_dir,t_mf+1,res)
    t_2 = np.ones((coord_2.shape[0], 1)) * ((t_mf+1) * 0.25 * T_scale)
    coord_3,vel_3,_,_,_,_ = dataprocess_MRI.preprocess_carotid_avg_2(data_dir,t_mf+2,res)
    t_3 = np.ones((coord_3.shape[0], 1)) * ((t_mf+2) * 0.25 * T_scale)

    coord_0 = (coord_0-L0) / L_gt
    coord_1 = (coord_1-L0) / L_gt
    coord_2 = (coord_2-L0) / L_gt
    coord_3 = (coord_3-L0) / L_gt
    coord = (coord-L0) / L_gt

    input_t_1 = np.concatenate((Scale_factor * coord_0,t_0),axis=1) 
    input_t0 = np.concatenate((Scale_factor * coord_1,t_1),axis=1)
    input_t1 = np.concatenate((Scale_factor * coord,t),axis=1)
    input_t2 = np.concatenate((Scale_factor * coord_2,t_2),axis=1)
    input_t3 = np.concatenate((Scale_factor * coord_3,t_3),axis=1)
    data_num_1 = len(input_t0)

    input_ts = np.concatenate((input_t_1, input_t0,input_t1,input_t2, input_t3),axis=0)
    vel_gt = np.concatenate((vel_0, vel_1,vel,vel_2, vel_3),axis=0)
    # input_ts = np.concatenate((input_t0,input_t1,input_t2),axis=0)
    # vel_gt = np.concatenate((vel_1,vel,vel_2),axis=0)

    gt = vel_gt / U_gt

    # print('coord shape : ', input_ts.shape)
    # print('vel shape : ', gt.shape)
    # print('L : ', L_gt)
    # print("U : ", U_gt)
    # print(data_num_1)

    # Initial Gaussian is downsampled from the tarin point
    coord = input_ts[::initial_gaussian_res]
    vel = gt[::initial_gaussian_res]
    print(f'Initial Gaussians : {coord.shape[0]}')

    
    mean = torch.tensor(coord.reshape(-1, 4), dtype=torch.float32, device=device, requires_grad=True)
    # P = torch.randn(mean.shape[0],1, device=device)
    P = torch.zeros(mean.shape[0], 1, dtype=torch.float32, device=device)
    zero_b = torch.unsqueeze(torch.tensor(np.zeros(len(mean)), dtype=torch.float32, requires_grad=True),-1).to(device)

    vel_ts = torch.tensor(vel.reshape(-1,3), dtype=torch.float32, device=device)

    dist = torch.ones_like(mean[:, 0]) * 10.0
    eps = torch.log(dist)[...,None].repeat(1, 4)
    prop = torch.concatenate([vel_ts, P], dim=1).requires_grad_()
    Gaussian_scale = (eps.detach()).requires_grad_()
    # Gaussian_rotation = torch.zeros((len(mean)),device=device).requires_grad_()

    #####
    input_tensor = torch.tensor(input_ts, dtype=torch.float32, device=device, requires_grad=True)

    # lr = 0.01
    # Z_thred = 0.0001
    # thr_dense = 0.002
    # thr_split = 2.0

    # epoch = 10000
    # batch_size = 1000
    criterion = torch.nn.MSELoss().to(device)

    Re = 1 / args.viscosity
    Re = U_gt * L_gt * Re 

    train_gt = torch.tensor(gt.reshape(-1,3), dtype=torch.float32, device=device)
    Model = Gaussian_PINN(
        Z_thred, mean, Gaussian_scale, prop, lr, thr_dense, thr_split).to(device)

    # Model = Gaussian_PINN(
    #     args, mean, Gaussian_scale, Gaussian_rotation, prop,lr, thr_dense, thr_split, alpha_shape).to(device)

    Model_dir = os.path.join(output_dir,'Weights')
    Mean_dir = os.path.join(output_dir,'Mean')
    Scale_dir = os.path.join(output_dir,'Scale')
    os.makedirs(Model_dir, exist_ok=True)
    os.makedirs(Mean_dir, exist_ok=True)
    os.makedirs(Scale_dir, exist_ok=True)

    ##### train in baches
    start_time = time.time()

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
        # "Rot" : Model.rot,
        "Scale" : Model.scale,
        "Property" : Model.property,
    }, path)
    print("Saved checkpoint at ", path)

    num_samples = input_tensor.shape[0]
    for iters in tqdm(range(1, epochs + 1)):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        Data_Loss = 0.0
        PDE_Loss = 0.0

        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]
            batch_input = input_tensor[indices]
            batch_gt = train_gt[indices]

            Model.optimizer.zero_grad()

            batch_input = batch_input.to(device).requires_grad_(True)
            pred = Model(batch_input)

            x, y, z, t = batch_input.split(1, dim=1)

            u = pred[:, 0:1]
            v = pred[:,1:2]
            w = pred[:, 2:3]
            p = pred[:,3:4]

            u_grad = torch.autograd.grad(u.sum(), batch_input, create_graph=True)[0]
            v_grad = torch.autograd.grad(v.sum(), batch_input, create_graph=True)[0]
            w_grad = torch.autograd.grad(w.sum(), batch_input, create_graph=True)[0]
            p_grad = torch.autograd.grad(p.sum(), batch_input, create_graph=True)[0]

            u_x,u_y,u_z,u_t = u_grad.split(1, dim=1)
            v_x,v_y,v_z,v_t = v_grad.split(1, dim=1)
            w_x,w_y,w_z,w_t = w_grad.split(1, dim=1)
            p_x,p_y,p_z,_   = p_grad.split(1, dim=1)
            
            ones = torch.ones_like(u_x)

            u_hess = torch.autograd.grad(u_x, batch_input, grad_outputs=ones, create_graph=True)[0]
            u_xx = u_hess[:, 0:1]
            u_yy = u_hess[:, 1:2]
            u_zz = u_hess[:, 2:3]

            v_hess = torch.autograd.grad(v_x, batch_input, grad_outputs=ones, create_graph=True)[0]
            v_xx = v_hess[:, 0:1]
            v_yy = v_hess[:, 1:2]
            v_zz = v_hess[:, 2:3]

            w_hess = torch.autograd.grad(w_x, batch_input, grad_outputs=ones, create_graph=True)[0]
            w_xx = w_hess[:, 0:1]
            w_yy = w_hess[:, 1:2]
            w_zz = w_hess[:, 2:3]

            Eu = u_t + u*u_x + v*u_y + w*u_z  - Scale_factor * (1/Re)*(u_xx + u_yy + u_zz) + p_x
            Ev = v_t + u*v_x + v*v_y + w*v_z  - Scale_factor * (1/Re)*(v_xx + v_yy + v_zz) + p_y
            Ew = w_t + u*w_x + v*w_y + w*w_z  - Scale_factor * (1/Re)*(w_xx + w_yy + w_zz) + p_z
            Ec = u_x + v_y + w_z

            # Eu = u_t/Scale_factor + u*u_x + v*u_y + w*u_z  - Scale_factor * (1/Re)*(u_xx + u_yy + u_zz) + p_x
            # Ev = v_t/Scale_factor + u*v_x + v*v_y + w*v_z  - Scale_factor * (1/Re)*(v_xx + v_yy + v_zz) + p_y
            # Ew = w_t/Scale_factor + u*w_x + v*w_y + w*w_z  - Scale_factor * (1/Re)*(w_xx + w_yy + w_zz) + p_z
            # Ec = u_x + v_y + w_z

            PDE_u, PDE_v, PDE_w, PDE_c = Eu.reshape(-1,1), Ev.reshape(-1,1), Ew.reshape(-1,1), Ec.reshape(-1,1)

            pt_Data_Loss = criterion(pred[:, :3], batch_gt)
            PDE_batch_Loss = (criterion(PDE_u, torch.zeros_like(PDE_u)) +
                                criterion(PDE_v, torch.zeros_like(PDE_v)) +
                                criterion(PDE_w, torch.zeros_like(PDE_w)) +
                                criterion(PDE_c, torch.zeros_like(PDE_c)))

            Data_batch_Loss = pt_Data_Loss.mean()

            batch_loss = Data_batch_Loss + PDE_batch_Loss
            batch_loss.backward()

            if iters%density_steps == 0 and iters > 100 :
                L_grad = Model.mean.grad

                diff = Model.calc_diff(batch_input)
                cov_inv = torch.diag_embed((-Model.scale).exp()).to('cuda')

                Z = Model.PDF(diff,cov_inv)
                Z_sum = torch.sum(Z,dim=1).unsqueeze(-1)
                W = Z / (Z_sum + 1e-10)

                pt_PDE_Loss = PDE_u.squeeze()**2 + PDE_v.squeeze()**2 + PDE_w.squeeze()**2 + PDE_c.squeeze()**2
                pt_total_loss = pt_Data_Loss + pt_PDE_Loss

                err = pt_total_loss.unsqueeze(-1)

                num = (W * err).sum(0)
                Gaussian_loss = num / (W.sum(0) + 1e-10)

            Model.optimizer.step()

            total_loss += batch_loss.item()
            Data_Loss += Data_batch_Loss.item()
            PDE_Loss += PDE_batch_Loss.item()

        # wandb.log({
        # "total_loss" : total_loss,
        # "PDE_loss" : PDE_Loss,
        # # "BDloss" : BD_Loss.item(),
        # "Data_loss" : Data_Loss,
        #     },step = iters)

        if iters % save_steps == 0:
            # print("Saving fig")
            Scale = Model.scale
            Plot_Scale = (Scale[:,0] + Scale[:,1]) / 2.0
            plt.scatter(Model.mean[:,0].cpu().detach(),Model.mean[:,1].cpu().detach(), c=Plot_Scale.data.cpu().numpy(), cmap='viridis')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.colorbar()
            fig_path = os.path.join(Scale_dir, '{:02d}.png'.format(iters))
            plt.savefig(fig_path)
            plt.close()
            
            # print("Saving model")
            path = os.path.join(Model_dir, '{:02d}.tar'.format(iters))
            torch.save({
                "Mean" : Model.mean,
                # "Rot" : Model.rot,
                "Scale" : Model.scale,
                "Property" : Model.property,
            }, path)
            print("Saved checkpoint at ", path)
        
        if iters%density_steps == 0 and iters > 100 :
            fig_path = os.path.join(Mean_dir, 'Mean{:02d}.png'.format(iters))
            Model.splating(L_grad, Gaussian_loss)
            
        if iters % Merge_step == 0:
            # Model.Gaussian_merge(pos_total)
            Model.Gaussian_merge(input_tensor)

        if iters%1 ==0 and False:
            print("Iteration: ",iters, \
                        "Data_loss_u: ",Data_Loss,"# Mean : ", len(Model.mean))

        if iters % 1000 == 0:
            print(f"iters: {iters}, {time.time() - start_time:.2f} sec")

    end_time = time.time()
    print(f"Excution time: {end_time - start_time:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data_dir', default='../Data/carotid/MRV_2.mat', type=str, help='Path to the training data file')
    parser.add_argument('--output_dir', default='./final_results_PINGS_avg', type=str, help='Base directory for saving results')
    parser.add_argument('--t_mf', default=12, type=int, help='Time frame for the main data')
    parser.add_argument('--res', default=2, type=int, help='Resolution for data preprocessing')

    parser.add_argument('--initial_gaussian_res', default=10, type=int, help='Downsampling rate for initial Gaussians')
    parser.add_argument('--scale_factor', default=100.0, type=float, help='Scale factor for input coordinates')

    parser.add_argument('--Z_thred', default=0.0001, type=float, help='0.01 / 0.001 / 0.0001 / 0.00001')
    
    parser.add_argument('--epochs', default=1000, type=int, help='10000 / 100000')
    parser.add_argument('--batch_size', default=10000, type=int, help='Batch size for training')

    parser.add_argument('--lr', default=0.01, type=float, help='5e-3 / 1e-3')
    parser.add_argument('--viscosity', type=float, default=4e-2, help='Viscosity of the fluid.')
    parser.add_argument('--save_step', default=100, type=int, help='100 / 1000')
    
    parser.add_argument('--dense_thred', default=0.0002, type=float, help='0.002 / 0.0002')
    parser.add_argument('--split_thred', default=2.0, type=float, help='2.0 / 1.0 / 0.0 / -1.0')
    parser.add_argument('--dense_step', default=100, type=int, help='100 / 500 / 10000')
    parser.add_argument('--merge_step', default=100, type=int, help='100 / 500 / 10000')
    args = parser.parse_args()
    main(args)

