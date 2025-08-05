import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import scipy.io
import pyvista as pv
import argparse

from utils import utils, dataprocess_MRI        
from PINN.model_4D import PINN                  

# reproducibility ----------------------------------------------------------
torch.manual_seed(777)
np.random.seed(777)

def RMSELoss(pred,target):
    # Loss = torch.sum((pred - target) ** 2, dim=1)
    # torch.mean((pred - target) ** 2, dim=1)
    Loss = np.mean((pred - target)**2)
    return np.sqrt(Loss)

def get_args():
    parser = argparse.ArgumentParser(description='Plot results from PINN/Siren models.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint file (.tar).')
    parser.add_argument('--model_type', type=str, default='Siren', choices=['PINN', 'Siren'], help='Type of the model to load.')
    parser.add_argument('--mat_path', type=str, default='../Data/carotid/MRV.mat', help='Path to the evaluation .mat file.')
    parser.add_argument('--training_path', type=str, default='../Data/carotid/MRV_2.mat', help='Path to the .mat file used for training normalization constants.')
    parser.add_argument('--t_frame', type=int, default=12, help='Time frame to evaluate (0-indexed).')
    parser.add_argument('--res_ds', type=int, default=1, help='Down-sampling interval for evaluation data.')
    return parser.parse_args()

def main():
    args = get_args()

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat_path    = args.mat_path
    training_path = args.training_path
    
    # ckpt_path   = f"./Results_carotid_PINN_avg_2/100000.tar"        
    # ckpt_path   = f"./Results_carotid_Siren_avg_2/100000.tar"
    ckpt_path = args.ckpt_path
    
    t_frame = args.t_frame
    res_ds = args.res_ds

    coord, vel_gt, L0, L1, L_gt, U_gt = dataprocess_MRI.preprocess_carotid_avg_2(
        mat_path, t_frame, res_ds)
    
    _, _, L0, L1, L_gt, U_gt = dataprocess_MRI.preprocess_carotid_avg_2(
        training_path, 12, 0)     


    T_scale   = U_gt / L_gt
    t_arr     = (t_frame * 0.25 * T_scale) * np.ones((coord.shape[0], 1))

    coord_nd  = (coord - L0) / L_gt
    vel_nd    = vel_gt / U_gt
    input_nd  = np.concatenate((coord_nd, t_arr), axis=1)

    # load
    activation = 'Tanh' if args.model_type == 'PINN' else 'Sin'
    model = PINN(4, 4, 256, 8, activation, 20, False, 10).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()                                 
    print(f"Loaded weight: {ckpt_path}")

    input_tensor = torch.tensor(input_nd, dtype=torch.float32,
                                requires_grad=False).to(device)
    gt_tensor    = torch.tensor(vel_nd,  dtype=torch.float32).to(device)

    batch_size   = 1024
    preds = []

    with torch.no_grad():
        for j in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[j:j+batch_size]
            x, y = batch[:, 0:1], batch[:, 1:2]
            z, t = batch[:, 2:3], batch[:, 3:4]
            pred = model(x, y, z, t)
            preds.append(pred[:, :3])             # u, v, w

    pred_tensor = torch.cat(preds, dim=0)

    rel_l2 = utils.relative_l2(pred_tensor, gt_tensor)
    print(f"\nRelative L2 Error (%): {100 * rel_l2.item():.2f}")

    vel_pred_phys = (pred_tensor * U_gt).cpu().numpy()   # (N,3)
    vel_gt_phys   = (gt_tensor   * U_gt).cpu().numpy()   # (N,3)
    print(f"RMSE : {RMSELoss(vel_pred_phys, vel_gt_phys):.2f}")


if __name__ == "__main__":
    main()
