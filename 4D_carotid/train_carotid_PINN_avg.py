import numpy as np 
import torch
import os 
import time
import matplotlib.pyplot as plt 

from utils import dataprocess_MRI,utils
from PINN import model_4D
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train PINN model on carotid artery data.')
    parser.add_argument('--data_dir', type=str, default='../Data/carotid/MRV_2.mat', help='Path to the training data file.')
    parser.add_argument('--model_dir', type=str, default='./Results_carotid_PINN_avg_2', help='Directory to save model weights and results.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu).')
    parser.add_argument('--t_mf', type=int, default=12, help='Time frame for the main data.')
    parser.add_argument('--res', type=int, default=2, help='Resolution for data preprocessing.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of training epochs.')
    parser.add_argument('--viscosity', type=float, default=4e-2, help='Viscosity of the fluid.')
    parser.add_argument('--save_interval', type=int, default=10000, help='Interval for saving model checkpoints.')
    parser.add_argument('--batch_size', default=10000, type=int, help='Batch size for training')
    # parser.add_argument('--wandb_project', type=str, default='ICCV_carotid', help='Wandb project name.')
    # parser.add_argument('--wandb_name', type=str, default='carotid_PINN_avg', help='Wandb run name.')
    return parser.parse_args()

def main():
    args = get_args()

    # wandb.login()
    # wandb.init(project="ICCV_carotid",
    #             name=f"carotid_PINN_avg",)

    device = args.device
    t_mf = args.t_mf

    data_dir =  args.data_dir
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

    input_t_1 = np.concatenate((coord_0,t_0),axis=1) 
    input_t0 = np.concatenate((coord_1,t_1),axis=1)
    input_t1 = np.concatenate((coord,t),axis=1)
    input_t2 = np.concatenate((coord_2,t_2),axis=1)
    input_t3 = np.concatenate((coord_3,t_3),axis=1)
    data_num_1 = len(input_t0)

    input_ts = np.concatenate((input_t_1, input_t0,input_t1,input_t2, input_t3),axis=0)
    vel_gt = np.concatenate((vel_0, vel_1,vel,vel_2, vel_3),axis=0)

    gt = vel_gt / U_gt
    # input_ts = (input_ts-L0) / L_gt

    # print('coord shape : ', input_ts.shape)
    # print('vel shape : ', gt.shape)
    # print('L : ', L_gt)
    # print("U : ", U_gt)
    # print(data_num_1)

    x = torch.tensor(input_ts[:,0].reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(input_ts[:,1].reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
    z = torch.tensor(input_ts[:,2].reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
    t = torch.tensor(input_ts[:,3].reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
    train_gt = torch.tensor(gt.reshape(-1,3), dtype=torch.float32).to(device)
    zero = torch.unsqueeze(torch.tensor(np.zeros(data_num_1), dtype=torch.float32, requires_grad=True),-1).to(device)

    lr = args.lr
    epochs = args.epochs
    Re = 1 / args.viscosity
    Re = U_gt * L_gt * Re 

    Model_2 = model_4D.PINN(4,4,256,8,'Tanh',20,False,10).to(device)
    optimizer = torch.optim.Adam(Model_2.parameters(), lr = lr)
    criterion = torch.nn.MSELoss().to(device)

    Model_dir = args.model_dir
    os.makedirs(Model_dir, exist_ok=True)

    start_time = time.time()
    utils.train_in_batches_PINN(Model_2,optimizer,x,y,z,t,data_num_1,train_gt, criterion, Re,L_gt,U_gt,epochs, args.batch_size, Model_dir)
    end_time = time.time()
    print(f"Excution time: {end_time - start_time:.2f} sec")

if __name__ == '__main__':
    main()
