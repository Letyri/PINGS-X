import torch
import math 
import numpy as np 
import matplotlib.pyplot as plt
import alphashape
import shapely.vectorized



def clipping_NS(mean, alpha_shape):
    mean_left = shapely.vectorized.contains(alpha_shape, mean[:, 0].cpu().detach(), mean[:, 1].cpu().detach())
    return mean_left

def clipping_Burgers(mean):
    mean_left = (mean[:,0] > -100.05) & (mean[:,0] < 100.05) & (mean[:,1] > - 0.05) & (mean[:,1] < 100.05)
    return mean_left 

def clipping_BB(mean):
    mean_left = (mean[:,0] > -40.05) & (mean[:,0] < 60.05) & (mean[:,1] > - 60.05) & (mean[:,1] < 40.05)    # x 4.0
    return mean_left 

def clipping_Lid(mean):
    mean_left = (mean[:,0] > -0.05) & (mean[:,0] < 100.05) & (mean[:,1] > - 0.05) & (mean[:,1] < 100.05)
    return mean_left 

def density_clone_split(data_type, mean,rot,scale,grad, Gaussian_loss, thres_grad,thres_split,wall:alphashape=None):
    
    print("------Densification------")
    
    det = torch.sqrt(grad[:,0]**2.0 + grad[:,1]**2.0)


    thres = Gaussian_loss.median() * 2.0
    pt_high = (Gaussian_loss > thres).squeeze()

    Split_density = ( (scale[:,0] + scale[:,1]) / 2.0 > thres_split) 
    Clone = pt_high & ~Split_density
    Split = pt_high & Split_density
    scale = scale.clone()
    
    ## Clone
    
    rot_clone = rot[Clone]
    scale_clone = scale[Clone] 
    mean_clone = mean[Clone] + scale[Clone].exp() * grad[Clone] / det[Clone].reshape(-1,1) 
    
    ## Split
    rot_Split = rot[Split]
    scale[Split] = scale[Split] - math.log(1.6)
    scale_Split = scale[Split] 
    mean_Split = mean[Split] + scale[Split].exp() * grad[Split] / det[Split].reshape(-1,1) 
    
    print("Clone Data shape")
    print(len(mean_clone))
    
    print("Split Data shape")
    print(len(mean_Split))
    
    
    mean_new = torch.concat((mean_clone,mean_Split),axis=0)
    rot_new = torch.concat((rot_clone,rot_Split),axis=0)
    scale_new = torch.concat((scale_clone,scale_Split),axis=0)
    

    if data_type=='bur':
        clip_mean = clipping_Burgers(mean_new) 
    elif data_type=='bb':
        clip_mean = clipping_BB(mean_new) 
    elif data_type=='lid':
        clip_mean = clipping_Lid(mean_new) 
    else:
        clip_mean = clipping_NS(mean_new,wall) 
        
    rot_new = rot_new[clip_mean]
    scale_new = scale_new[clip_mean]
    mean_new = mean_new[clip_mean]
    
    print("After Clipping shape")
    print(mean_new.shape)
    print("------------------------")
    
    
    
    return (mean_new.detach()).requires_grad_(),\
        (rot_new.detach()).requires_grad_()\
        ,(scale_new.detach()).requires_grad_(),\
        (scale.detach())