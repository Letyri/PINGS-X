import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import scipy.io
import pyvista as pv
import argparse

from utils import utils, dataprocess_MRI        
from Model.PINGS import Gaussian_PINN
# from Model.PINGS_wo_norm import Gaussian_PINN
# from Model.PINGS_wo_mean import Gaussian_PINN

# reproducibility ----------------------------------------------------------
torch.manual_seed(777)
np.random.seed(777)

def RMSELoss(pred,target):
    # Loss = torch.sum((pred - target) ** 2, dim=1)
    # torch.mean((pred - target) ** 2, dim=1)
    Loss = np.mean((pred - target)**2)
    return np.sqrt(Loss)

def get_args():
    parser = argparse.ArgumentParser(description='Plot results from PINGS-X model.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint file (.tar).')
    parser.add_argument('--mat_path', type=str, default='../Data/carotid/MRV.mat', help='Path to the evaluation .mat file.')
    parser.add_argument('--training_path', type=str, default='../Data/carotid/MRV_2.mat', help='Path to the .mat file used for training normalization constants.')
    parser.add_argument('--t_frame', type=int, default=12, help='Time frame to evaluate (0-indexed).')
    parser.add_argument('--res_ds', type=int, default=1, help='Down-sampling interval for evaluation data.')
    parser.add_argument('--scale_factor', type=float, default=100.0, help='Scale factor used during training.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')

    parser.add_argument('--Z_thred', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--thr_dense', type=float, default=2e-4)
    parser.add_argument('--thr_split', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = get_args()

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat_path    = args.mat_path
    training_path = args.training_path

    # ckpt_path = "./final_results_PINGS_avg_2_10_100_100_2/Weights/1000.tar"
    ckpt_path = args.ckpt_path

    t_frame     = args.t_frame
    res_ds      = args.res_ds

    Z_thred     = args.Z_thred
    lr          = args.lr
    thr_dense   = args.thr_dense
    thr_split   = args.thr_split
    Scale_factor = args.scale_factor

    coord, vel_gt, L0, _, L_gt, U_gt = dataprocess_MRI.preprocess_carotid_avg_2(
        mat_path, t_frame, res_ds)

    _, _, L0, _, L_gt, U_gt = dataprocess_MRI.preprocess_carotid_avg_2(
        training_path, 12, 0)

    T_scale   = U_gt / L_gt
    t_arr     = (t_frame * 0.25 * T_scale) * np.ones((coord.shape[0], 1))

    coord_nd  = (coord - L0) / L_gt
    vel_nd    = vel_gt / U_gt
    input_nd  = np.concatenate((Scale_factor * coord_nd, t_arr), axis=1)

    # load
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded keys : {list(ckpt.keys())}")

    mean = ckpt['Mean'].to(device).clone().detach().requires_grad_(True)
    scale = ckpt['Scale'].to(device).clone().detach().requires_grad_(True)
    prop = ckpt['Property'].to(device).clone().detach().requires_grad_(True)
    model = Gaussian_PINN(Z_thred, mean, scale, prop,
                    lr, thr_dense, thr_split).to(device)
    model.eval()                                
    print(f"Loaded weight: {ckpt_path}")
    print(f"(Gaussians 개수 : {len(mean)})")

    input_tensor = torch.tensor(input_nd, dtype=torch.float32,
                                requires_grad=False).to(device)
    gt_tensor    = torch.tensor(vel_nd,  dtype=torch.float32).to(device)

    batch_size   = 1024
    preds = []

    with torch.no_grad():
        for j in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[j:j+batch_size]
            pred = model(batch)
            preds.append(pred[:, :3])             # u, v, w

    pred_tensor = torch.cat(preds, dim=0)

    rel_l2 = utils.relative_l2(pred_tensor, gt_tensor)
    print(f"\nRelative L2 Error (%) : {100 * rel_l2.item():.2f}")

    vel_pred_phys = (pred_tensor * U_gt).cpu().numpy()   # (N,3)
    vel_gt_phys   = (gt_tensor   * U_gt).cpu().numpy()   # (N,3)
    print(f"RMSE : {RMSELoss(vel_pred_phys, vel_gt_phys):.2f}")


if __name__ == "__main__":
    main()
