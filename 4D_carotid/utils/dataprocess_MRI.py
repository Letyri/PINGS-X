import numpy as np 
import torch
import scipy.io

from torch.autograd import Variable
torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)


device = 'cuda'

# load spatial averaged data
def preprocess_carotid_avg_2(data_dir: str, t: int, res: int = 2):
    mat = scipy.io.loadmat(data_dir, squeeze_me=True)
    x, y, z = mat['x'], mat['y'], mat['z']
    u, v, w = mat['U'][t], mat['V'][t], mat['W'][t]
    
    coord = np.stack((x, y, z), axis=-1)
    vel = np.stack((u, v, w), axis=-1)

    L0 = np.min(coord)
    L1 = np.max(coord)
    L = L1 - L0
    U = np.max(vel)
    
    return coord, vel, L0, L1, L, U
