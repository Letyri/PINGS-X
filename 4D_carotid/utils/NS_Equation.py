import numpy as np
import torch

device = 'cuda'
criterion = torch.nn.MSELoss().to(device)


def L_NS_pinn_PDE_4D(x,y,z,t,model,Re,L,U):
    sol = model(x,y,z,t)
    u = sol[:,0].reshape(-1,1)
    v = sol[:,1].reshape(-1,1)
    w = sol[:,2].reshape(-1,1)
    p = sol[:,3].reshape(-1,1)
    
    u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.reshape(-1,1).sum(), z, create_graph=True)[0]
    u_t = torch.autograd.grad(u.reshape(-1,1).sum(), t, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
    v_z = torch.autograd.grad(v.reshape(-1,1).sum(), z, create_graph=True)[0]
    v_t = torch.autograd.grad(v.reshape(-1,1).sum(), t, create_graph=True)[0]
    
    w_x = torch.autograd.grad(w.reshape(-1,1).sum(), x, create_graph=True)[0]
    w_y = torch.autograd.grad(w.reshape(-1,1).sum(), y, create_graph=True)[0]
    w_z = torch.autograd.grad(w.reshape(-1,1).sum(), z, create_graph=True)[0]
    w_t = torch.autograd.grad(w.reshape(-1,1).sum(), t, create_graph=True)[0]
    
    p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
    p_z = torch.autograd.grad(p.reshape(-1,1).sum(), z, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones(z.shape).to(device), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, torch.ones(z.shape).to(device), create_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, torch.ones(z.shape).to(device), create_graph=True)[0]
    
    # Eu = (L/(3.0*U))*u_t + u*u_x + v*u_y + w*u_z  - (1/Re)*(u_xx + u_yy + u_zz) + p_x
    # Ev = (L/(3.0*U))*v_t + u*v_x + v*v_y + w*v_z  - (1/Re)*(v_xx + v_yy + v_zz) + p_y
    # Ew = (L/(3.0*U))*w_t + u*w_x + v*w_y + w*w_z  - (1/Re)*(w_xx + w_yy + w_zz) + p_z
    Eu = u_t + u*u_x + v*u_y + w*u_z  - (1/Re)*(u_xx + u_yy + u_zz) + p_x
    Ev = v_t + u*v_x + v*v_y + w*v_z  - (1/Re)*(v_xx + v_yy + v_zz) + p_y
    Ew = w_t + u*w_x + v*w_y + w*w_z  - (1/Re)*(w_xx + w_yy + w_zz) + p_z
    Ec = u_x + v_y + w_z
    
    return sol,Eu.reshape(-1,1),Ev.reshape(-1,1),Ew.reshape(-1,1),Ec.reshape(-1,1)