import numpy as np
import torch
import networkx as nx 
import os
import wandb 
from tqdm import tqdm

from torch.autograd import Variable
from utils import NS_Equation

device = 'cuda'

def relative_l2(u, u_gt):
    return torch.norm(u - u_gt) / torch.norm(u_gt)

def MSELoss(pred,target):
    Loss = torch.sum((pred - target) ** 2, dim=1)
    return Loss 
    
def train_in_batches_PINN(model,optimizer,x,y,z,t,data_num,train_gt, criterion, Re,L,U, epochs, batch_size,Model_dir):
    num_samples = x.shape[0]
    
    for epoch in tqdm(range(1, epochs + 1)):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        Data_Loss = 0.0
        PDE_Loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]  
            batch_input_x = x[indices]
            batch_input_y = y[indices]
            batch_input_z = z[indices]
            batch_input_t = t[indices]
            
            batch_gt = train_gt[indices]
            
            optimizer.zero_grad()
            pred,PDE_u,PDE_v,PDE_w,PDE_c = NS_Equation.L_NS_pinn_PDE_4D(batch_input_x,batch_input_y,batch_input_z,batch_input_t,model,Re,L,U)
            # pred = model(batch_input_x,batch_input_y,batch_input_z,batch_input_t)   # w/o PDE loss
            
            Data_batch_Loss = criterion(pred[:, :3], batch_gt)
            PDE_batch_Loss = (criterion(PDE_u, torch.zeros_like(PDE_u)) +
                        criterion(PDE_v, torch.zeros_like(PDE_v)) +
                        criterion(PDE_w, torch.zeros_like(PDE_w)) +
                        criterion(PDE_c, torch.zeros_like(PDE_c)))
            
            batch_loss = Data_batch_Loss + PDE_batch_Loss
            # batch_loss = Data_batch_Loss  # w/o PDE loss
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            Data_Loss += Data_batch_Loss.item()
            PDE_Loss += PDE_batch_Loss.item() 
            
        
        if epoch%100 ==0:
            print("Saving model")
            path = os.path.join(Model_dir, '{:02d}.tar'.format(epoch))
            torch.save({
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }, path)
            print("Saved checkpoint at ", path)
        
        # wandb.log({
        #         "Loss" : total_loss,
        #         "PDEloss" : PDE_Loss,
        #         "Valid" : Data_Loss,
        #             },step = epoch)
        
        
        if epoch % 50 == 0 :
            print(f"Epoch {epoch}, Data Loss: {Data_Loss:.6f}, PDE Loss: {PDE_Loss:.6f}")
