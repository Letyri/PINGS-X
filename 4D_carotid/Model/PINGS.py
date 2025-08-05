import numpy as np 
import torch

from utils.Gaussian import *  
from utils.Density import * 
from utils.utils import *  
from torch import nn

class Gaussian_PINN(nn.Module):
    def setup_func(self):
        def calc_diff(input):
            diff = input.unsqueeze(1) - self.mean.unsqueeze(0)
            
            return diff
        
        def pdf(diff,cov_inv):
            diff_cov = torch.einsum('abj,bjk->abk', diff, cov_inv)  # (A, B, 3) @ (3, 3) = (A, B, 3)
            Z = torch.sum(diff_cov * diff, dim=-1)  # (A, B)
            Z = torch.exp(-0.5 * Z)  # (A, B)
            
            Z = Z * (Z > self.Z_thr).float()  # filtering
            
            return Z
        
        def diff_nm1(diff,cov_inv,diff_cov):
            Z = pdf(diff,cov_inv)
            Z_sum = torch.sum(Z, dim=1, keepdim=True) +1e-6  # (10436, 1)
            
            dZ_dinput = -Z.unsqueeze(-1) * diff_cov  # (10436, 400, 1) * (10436, 400, 2) -> (10436, 400, 2)
            sum_dZ_dinput = torch.sum(dZ_dinput, dim=1, keepdim=True)  # (10436, 1, 2)
            dZ_norm_dinput = (dZ_dinput * Z_sum.unsqueeze(-1) - Z.unsqueeze(-1) * sum_dZ_dinput) / (Z_sum.unsqueeze(-1) ** 2)  # (10436, 400, 2)
            
            return dZ_norm_dinput
        
        def diff_nm2(diff,cov_inv,diff_cov):
            Z = pdf(diff,cov_inv)
            Z_sum = torch.sum(Z, dim=1, keepdim=True) +1e-6  # (10436, 1)
            
            term2 = torch.matmul(diff_cov.unsqueeze(-2), cov_inv.unsqueeze(0))  # (10436, 400, 1, 2) x (1, 400, 2, 2) -> (10436, 400, 2, 2)
            d2Z_dinput2 = Z.unsqueeze(-1).unsqueeze(-1) * (term2 - diff_cov.unsqueeze(-2) * diff_cov.unsqueeze(-1))  # (10436, 400, 1, 1) * ((10436, 400, 2, 2) - (10436, 400, 2, 1) * (10436, 400, 1, 2)) -> (10436, 400, 2, 2)
            
            # Normalize second derivative
            sum_d2Z_dinput2 = torch.sum(d2Z_dinput2, dim=1, keepdim=True)  # (10436, 1, 2, 2)
            d2Z_norm_dinput2 = (d2Z_dinput2 * Z_sum.unsqueeze(-1).unsqueeze(-1) - Z.unsqueeze(-1).unsqueeze(-1) * sum_d2Z_dinput2) / (Z_sum.unsqueeze(-1).unsqueeze(-1) ** 2)  # (10436, 400, 2, 2)
            
            return d2Z_norm_dinput2
        
        def calc_pred(Z):
            # normalization
            Z_sum = torch.sum(Z,dim=1).unsqueeze(-1)
            Z = Z / (Z_sum + 1e-6)
            Pred = torch.sum(self.property.unsqueeze(0) * Z.unsqueeze(-1),dim=1)
            
            return Pred
        
        self.calc_diff = calc_diff
        self.diff_nm1 = diff_nm1
        self.diff_nm2 = diff_nm2
        self.PDF = pdf
        self.calc_pred = calc_pred
        
    # Adam optimizer
    def Set_Optimizer(self,lr:float=1e-4):
        l = [
            {'params': [self.mean], 'lr': lr, "name": "Mean"},
            {'params': [self.scale], 'lr': lr, "name": "Scale"},
            {'params': [self.property], 'lr': lr, "name": "Property"},
            ]
        
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def __init__(self,Z_thred,mean,scale,property,
                 lr:float=1e-4,
                 thres_dense:float=0.0002,thres_split:float=-1.0,
                 ):
       super().__init__()
       self.Z_thr = Z_thred
       self.mean = mean
       self.scale = scale 
       self.property = property
       
       self.lr = lr 
       self.optimizer = self.Set_Optimizer(lr)
       
       #Threshold scale 
       self.thr_dense = thres_dense
       self.thr_split = thres_split
       
       self.setup_func()
    
    def forward(self,input):
        diff = self.calc_diff(input)
        cov_inv = torch.diag_embed((-self.scale).exp()).to('cuda')
        
        Z = self.PDF(diff,cov_inv)
        output = self.calc_pred(Z)
            
        return output  
    
    def NS_Equation(self,input,Re):
        diff = self.calc_diff(input)
        cov_inv = torch.diag_embed((-self.scale).exp()).to('cuda')
        diff_cov = torch.einsum('abj,bjk->abk', diff, cov_inv)
        
        property_u = self.property[:, 0]
        property_v = self.property[:, 1]
        property_w = self.property[:, 2]
        property_p = self.property[:, 3]
        
        Z = self.PDF(diff,cov_inv)
        Z_1 = self.diff_nm1(diff,cov_inv,diff_cov)
        Z_2 = self.diff_nm2(diff,cov_inv,diff_cov)
        Z_2_diag = Z_2.diagonal(dim1=-2, dim2=-1) 
       
        u,v,w,p = self.calc_pred(Z).unbind(dim=-1)
        u_x,u_y,u_z = torch.einsum("bij,i->bj", Z_1,property_u).unbind(dim=-1)
        v_x,v_y,v_z = torch.einsum("bij,i->bj", Z_1,property_v).unbind(dim=-1)
        w_x,w_y,w_z = torch.einsum("bij,i->bj", Z_1,property_w).unbind(dim=-1)
        p_x,p_y,p_z = torch.einsum("bij,i->bj", Z_1,property_p).unbind(dim=-1)
        u_xx,u_yy,u_zz = torch.einsum("bij,i->bj", Z_2_diag,property_u).unbind(dim=-1)
        v_xx,v_yy,v_zz = torch.einsum("bij,i->bj", Z_2_diag,property_v).unbind(dim=-1)
        w_xx,w_yy,w_zz = torch.einsum("bij,i->bj", Z_2_diag,property_w).unbind(dim=-1)
        
        Eu = u*u_x + v*u_y + w*u_z  - (1/Re)*(u_xx + u_yy + u_zz) + p_x
        Ev = u*v_x + v*v_y + w*v_z  - (1/Re)*(v_xx + v_yy + v_zz) + p_y
        Ew = u*w_x + v*w_y + w*w_z  - (1/Re)*(w_xx + w_yy + w_zz) + p_z
        Ec = u_x + v_y + w_z
        
        return Eu.reshape(-1,1),Ev.reshape(-1,1),Ew.reshape(-1,1),Ec.reshape(-1,1)
    
    def Remove_params(self,Idx):
        print("------Removing------")
        print("------------------------")
        
        print("Previous Data shape")
        print(self.mean.shape)
        
        Idx = list(set(range(len(self.mean))) - set(Idx))
        
        self.mean = self.mean[Idx]
        self.scale = self.scale[Idx]
        self.property = self.property[Idx]
        
        print("After Data shape")
        print(self.mean.shape)
        
        for group in self.optimizer.param_groups:
                if group['name'] == 'Mean':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = stored_state["exp_avg"][Idx]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][Idx]
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.mean
                    self.optimizer.state[group['params'][0]] = stored_state

                if group['name'] == 'Scale':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = stored_state["exp_avg"][Idx]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][Idx]
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.scale
                    self.optimizer.state[group['params'][0]] = stored_state
                    
                if group['name'] == 'Property':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = stored_state["exp_avg"][Idx]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][Idx]
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.property
                    self.optimizer.state[group['params'][0]] = stored_state
                          
    def Add_params(self,mean,scale,property):
        self.mean = (torch.concat((self.mean,mean),axis=0).detach()).requires_grad_()
        self.scale = (torch.concat((self.scale,scale),axis=0).detach()).requires_grad_()
        self.property = (torch.concat((self.property,property),axis=0).detach()).requires_grad_()
        
        for group in self.optimizer.param_groups:
                if group['name'] == 'Mean':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(scale)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(scale)), dim=0)
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.mean
                    self.optimizer.state[group['params'][0]] = stored_state

                if group['name'] == 'Scale':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(scale)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(scale)), dim=0)
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.scale
                    self.optimizer.state[group['params'][0]] = stored_state
                    
                if group['name'] == 'Property':
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(property)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(property)), dim=0)
                    
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = self.property
                    self.optimizer.state[group['params'][0]] = stored_state
        
    def Gaussian_merge(self,input):
        diff = self.calc_diff(input)
        cov_inv = torch.diag_embed((-self.scale).exp()).to('cuda')
        Z = self.PDF(diff,cov_inv)
        S_matrix = compute_cosine_similarity_matrix(Z) - torch.eye(len(self.mean),device=self.mean.device)
        indices = torch.where(torch.triu(S_matrix) > 0.9)
        index_tensor = torch.stack(indices, dim=1)
        Cluster = make_clusters(index_tensor)
        
        mean_new = torch.zeros((len(Cluster),self.mean.shape[1]),device=self.mean.device)
        scale_new = torch.zeros((len(Cluster),self.scale.shape[1]),device=self.mean.device)
        property_new = torch.zeros((len(Cluster),self.property.shape[1]),device=self.mean.device)
        
        Remove_index = []
        index = 0
        
        for cluster in Cluster:
            mean_new[index] = self.mean[cluster].mean(dim=0, keepdim=True)
            scale_new[index] = self.scale[cluster].mean(dim=0, keepdim=True)
            property_new[index] = self.forward(mean_new[index].reshape(1,-1))
            r_index = cluster
            Remove_index.extend(r_index)
            index = index+1
        
        self.Remove_params(Remove_index)
        self.Add_params((mean_new.detach()).requires_grad_()\
                        ,(scale_new.detach()).requires_grad_()\
                        ,(property_new.detach()).requires_grad_())
        
    def splating(self,L_grad, Gaussian_loss):
        mean_new,scale_new,self.scale = density_clone_split(
            self.mean,self.scale,
            L_grad,
            Gaussian_loss,
            self.thr_dense,
            self.thr_split)
        property_new = ((self.forward(mean_new)).detach()).requires_grad_()
        self.Add_params(mean_new,scale_new,property_new)
              
    