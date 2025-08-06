import numpy as np 
import torch
import shapely.vectorized

from utils.Gaussian import *  
from utils.Density import * 
from utils.utils import *  
from torch import nn

class Gaussian_PINN(nn.Module):
    def setup_func(self):
        self.I = torch.eye(2).to("cuda")
        
        def Gaussian_diff_NM(input):
            diff = input.unsqueeze(1) - self.mean.unsqueeze(0)
            cov_inv = calc_covariance(self.rot,self.scale)
            diff_cov_inv = torch.matmul(diff.unsqueeze(-2), cov_inv).squeeze(-2)  # (10436, 400, 1, 2) x (400, 2, 2) -> (10436, 400, 1, 2) -> (10436, 400, 2)
            Z = torch.exp(-0.5 * torch.sum(diff_cov_inv * diff, dim=-1))  # (10436, 400)
            Z = torch.where(Z > self.Z_thr, Z, torch.zeros_like(Z))
            Z_sum = torch.sum(Z, dim=1, keepdim=True) +1e-6  # (10436, 1)
            Z_norm = Z / (Z_sum)  # (10436, 400)
            
            dZ_dinput = -Z.unsqueeze(-1) * diff_cov_inv  # (10436, 400, 1) * (10436, 400, 2) -> (10436, 400, 2)
            sum_dZ_dinput = torch.sum(dZ_dinput, dim=1, keepdim=True)  # (10436, 1, 2)
            dZ_norm_dinput = (dZ_dinput * Z_sum.unsqueeze(-1) - Z.unsqueeze(-1) * sum_dZ_dinput) / (Z_sum.unsqueeze(-1) ** 2)  # (10436, 400, 2)
            
            term1 = diff_cov_inv  # (10436, 400, 2)
            term2 = torch.matmul(term1.unsqueeze(-2), cov_inv.unsqueeze(0))  # (10436, 400, 1, 2) x (1, 400, 2, 2) -> (10436, 400, 2, 2)
            
            # We need to account for the product rule here
            d2Z_dinput2 = Z.unsqueeze(-1).unsqueeze(-1) * (term2 - term1.unsqueeze(-2) * term1.unsqueeze(-1))  # (10436, 400, 1, 1) * ((10436, 400, 2, 2) - (10436, 400, 2, 1) * (10436, 400, 1, 2)) -> (10436, 400, 2, 2)
            # Normalize second derivative
            sum_d2Z_dinput2 = torch.sum(d2Z_dinput2, dim=1, keepdim=True)  # (10436, 1, 2, 2)
            d2Z_norm_dinput2 = (d2Z_dinput2 * Z_sum.unsqueeze(-1).unsqueeze(-1) - Z.unsqueeze(-1).unsqueeze(-1) * sum_d2Z_dinput2) / (Z_sum.unsqueeze(-1).unsqueeze(-1) ** 2)  # (10436, 400, 2, 2)
            
            return Z_norm, dZ_norm_dinput, d2Z_norm_dinput2
        
        def calc_covariance(rot,scale):
            matrix_rot1,matrix_rot2 = Cal_rot(rot)
            matrix_scale = torch.diag_embed((-scale).exp()).to('cuda')
            cov_1 = torch.bmm(matrix_rot1,matrix_scale)
            matrix_cov = torch.bmm(cov_1,matrix_rot2)
            
            return matrix_cov
        
        def pdf(input):
            diff = input.unsqueeze(1) - self.mean.unsqueeze(0)
            cov = calc_covariance(self.rot,self.scale)
            Z = torch.exp(-1/2 * torch.einsum('nij,nij->ni', torch.einsum('nij,ijk->nik',diff,cov),diff))
            Z = torch.where(Z > self.Z_thr, Z, torch.zeros_like(Z))
            
            return Z
               
        def prediction(Z):
            Z_sum = torch.sum(Z,dim=1).unsqueeze(-1)
            Z = Z / (Z_sum + 1e-6)
            Pred = torch.sum(self.property.unsqueeze(0) * Z.unsqueeze(-1),dim=1)
            
            return Pred
        
        self.cov = calc_covariance
        self.diff_nm = Gaussian_diff_NM
        self.PDF = pdf
        self.Pred = prediction
    
    # Adam optimizer
    def Set_Optimizer(self,lr:float=1e-4):
        l = [
            # {'params': [self.rot], 'lr': lr, "name": "Rot"},
            {'params': [self.mean], 'lr': lr, "name": "Mean"},
            {'params': [self.scale], 'lr': lr, "name": "Scale"},
            {'params': [self.property], 'lr': lr, "name": "Property"},
            ]
        
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def __init__(self,args,mean,scale,rot,property,
                 lr:float=1e-4,
                 thres_dense:float=0.0002,thres_split:float=-1.0,
                 alphashape:alphashape=None):
       super().__init__()
       self.args = args
       self.Z_thr = self.args.Z_thred
       self.mean = mean
       self.scale = scale 
       self.rot = rot
       self.property = property
       
       self.lr = lr 
       self.optimizer = self.Set_Optimizer(lr)
       
       #Threshold scale 
       self.thr_dense = thres_dense
       self.thr_split = thres_split
       
       self.alpha_shape = alphashape
       self.setup_func()
    
    def Remove_params(self,Idx):
        print("------Removing------")
        print("------------------------")
        
        print("Previous Data shape")
        print(self.mean.shape)
        
        Idx = list(set(range(len(self.mean))) - set(Idx))
        
        self.mean = self.mean[Idx]
        self.rot = self.rot[Idx]
        self.scale = self.scale[Idx]
        self.property = self.property[Idx]
        
        print("After Data shape")
        print(self.mean.shape)
        
        for group in self.optimizer.param_groups:
                # if group['name'] == 'Rot':
                #     stored_state = self.optimizer.state.get(group['params'][0], None)
                #     stored_state["exp_avg"] = stored_state["exp_avg"][Idx]
                #     stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][Idx]
                    
                #     del self.optimizer.state[group['params'][0]]
                #     group["params"][0] = self.rot
                #     self.optimizer.state[group['params'][0]] = stored_state
                    
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
                          
    def Add_params(self,mean,rot,scale,property):
        self.mean = (torch.concat((self.mean,mean),axis=0).detach()).requires_grad_()
        self.rot = (torch.concat((self.rot,rot),axis=0).detach()).requires_grad_()
        self.scale = (torch.concat((self.scale,scale),axis=0).detach()).requires_grad_()
        self.property = (torch.concat((self.property,property),axis=0).detach()).requires_grad_()
        
        for group in self.optimizer.param_groups:
                # if group['name'] == 'Rot':
                #     stored_state = self.optimizer.state.get(group['params'][0], None)
                #     stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(rot)), dim=0)
                #     stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(rot)), dim=0)
                    
                #     del self.optimizer.state[group['params'][0]]
                #     group["params"][0] = self.rot
                #     self.optimizer.state[group['params'][0]] = stored_state

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
        Z = self.PDF(input)
        S_matrix = compute_cosine_similarity_matrix(Z) - torch.eye(len(self.mean),device=self.mean.device)
        indices = torch.where(torch.triu(S_matrix) > 0.9)
        index_tensor = torch.stack(indices, dim=1)
        Cluster = make_clusters(index_tensor)
        
        mean_new = torch.zeros((len(Cluster),2),device=self.mean.device)
        rot_new = torch.zeros((len(Cluster)),device=self.mean.device)
        scale_new = torch.zeros((len(Cluster),2),device=self.mean.device)
        property_new = torch.zeros((len(Cluster),self.property.shape[1]),device=self.mean.device)
        
        Remove_index = []
        index = 0
        
        for cluster in Cluster:
            mean_new[index] = self.mean[cluster].mean(dim=0, keepdim=True)
            scale_new[index] = self.scale[cluster].mean(dim=0, keepdim=True)
            rot_new[index] = self.rot[cluster].mean(dim=0, keepdim=True)
            property_new[index] = self.forward(mean_new[index].reshape(1,-1))

            r_index = cluster
            Remove_index.extend(r_index)
            index = index+1
        
        self.Remove_params(Remove_index)
        self.Add_params((mean_new.detach()).requires_grad_()\
                        ,(rot_new.detach()).requires_grad_()\
                        ,(scale_new.detach()).requires_grad_()\
                        ,(property_new.detach()).requires_grad_())
        
    def splating(self,L_grad, Gaussian_loss, fig_path):
        mean_new,rot_new,scale_new,self.scale = density_clone_split(
            self.args.data_type,
            self.mean,self.rot,self.scale,
            L_grad,
            Gaussian_loss,
            self.thr_dense,
            self.thr_split,
            self.alpha_shape)
        property_new = ((self.forward(mean_new)).detach()).requires_grad_()
        
        plt.scatter(self.mean[:,0].cpu().detach(),self.mean[:,1].cpu().detach(), c='b')
        plt.scatter(mean_new[:,0].cpu().detach(),mean_new[:,1].cpu().detach(), c='r')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(fig_path)
        plt.close()
        
        self.Add_params(mean_new,rot_new,scale_new,property_new)
        
    def Plot_individual_GS(self,input,output_dir):
        Z = self.PDF(input)
        import os
        for i in range(len(self.mean)):
            plt.scatter(input[:,0].cpu().detach(),input[:,1].cpu().detach(), c=Z[:,i].data.cpu().numpy(), cmap='viridis')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.colorbar()
            fig_path = os.path.join(output_dir, '{:02d}.png'.format(i))
            plt.savefig(fig_path)
            plt.close()
            
    def Teaser_GS(self,input,output_dir):
        Z = self.PDF(input)
        Pred = self.property.unsqueeze(0) * Z.unsqueeze(-1)
        plt.figure()
        print(Pred.shape)
        import os
        for i in range(len(self.mean)):
            mask = torch.abs(Z[:, i]) > 0.90
            mask = mask.reshape(-1)
            plt.scatter(input[mask,0].cpu().detach(),input[mask,1].cpu().detach(),
                        alpha=0.3, 
                        c=Pred[mask,i].data.cpu().numpy(), cmap='viridis',
            vmin=-1, vmax=1)
            
            
        fig_path = os.path.join(output_dir, 'Teaser.png')           
        plt.savefig(fig_path)
        plt.close()    
            
    def forward(self,input):
        Z = self.PDF(input)
        output = self.Pred(Z)
            
        return output  
