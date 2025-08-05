import torch
import math 
import numpy as np 
import matplotlib.pyplot as plt



def clipping_Lid(mean):
    mean_left = (mean[:,0] > -0.05) & (mean[:,0] < 64.05) & (mean[:,1] > - 0.05) & (mean[:,1] < 25.05) & (mean[:,2] > - 0.05) & (mean[:,2] < 25.05)
    return mean_left 

def density_clone_split(mean,scale,grad, Gaussian_loss, thres_grad,thres_split):
    
    print("------Densification------")
    
    det = torch.sqrt(grad[:,0]**2.0 + grad[:,1]**2.0 + grad[:,2]**2.0)
    # thres_grad = 3.0 * det.mean().item()
    # pt_high = (det > thres_grad).squeeze()
    thres = Gaussian_loss.median() * 2.0
    pt_high = (Gaussian_loss > thres).squeeze()

    Split_density = ( (scale[:,0] + scale[:,1] + scale[:,2]) / 3.0 > thres_split) 
    Clone = pt_high & ~Split_density
    Split = pt_high & Split_density
    scale = scale.clone()
    
    ## Clone
    
    scale_clone = scale[Clone] 
    mean_clone = mean[Clone] + scale[Clone].exp() * grad[Clone] / det[Clone].reshape(-1,1) 
    
    ## Split
    scale[Split] = scale[Split] - math.log(1.6)
    scale_Split = scale[Split] 
    mean_Split = mean[Split] + scale[Split].exp() * grad[Split] / det[Split].reshape(-1,1) 
    
    print("Clone Data shape")
    print(len(mean_clone))
    
    print("Split Data shape")
    print(len(mean_Split))
    
    
    mean_new = torch.concat((mean_clone,mean_Split),axis=0)
    scale_new = torch.concat((scale_clone,scale_Split),axis=0)
    
    clip_mean = clipping_Lid(mean_new)
        
    scale_new = scale_new[clip_mean]
    mean_new = mean_new[clip_mean]
    
    print("After Clipping shape")
    print(mean_new.shape)
    print("------------------------")
    
    return (mean_new.detach()).requires_grad_(),\
        (scale_new.detach()).requires_grad_(),\
        (scale.detach())

