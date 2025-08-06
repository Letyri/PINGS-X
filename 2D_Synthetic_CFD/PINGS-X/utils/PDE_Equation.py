import numpy as np
import torch

device = 'cuda'
criterion = torch.nn.MSELoss().to(device)

def L_bur_Gaussian_PDE(input,Model,Scale):
    P_0,P_1,P_2 = Model.diff_nm(input) # (input, #mean ,) , (input, #mean,2) , (input, #mean,2,2)
    property_0 = Model.property[:, 0].unsqueeze(0)
    
    u = torch.sum(property_0 * P_0, dim=1)
    u_x = torch.sum(property_0 * P_1[:,:,0], dim=1)
    u_t = torch.sum(property_0 * P_1[:,:,1], dim=1)
    u_xx = torch.sum(property_0 * P_2[:,:,0,0], dim=1)
    
    f = Burgers(u,
                u_x,u_t,
                u_xx*Scale)
    
    return f.reshape(-1,1)

def L_bb_Gaussian_PDE_3(input,Model,Scale1,Scale2):
    x, y = input[:, 0].unsqueeze(-1), input[:, 1].unsqueeze(-1)
    sol = Model(torch.concat((x, y),axis=1))

    u = sol[:,0].reshape(-1,1)
    v = sol[:,1].reshape(-1,1)
    
    u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
    
    u_t = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    
    f1 = BB1(u, u_t * Scale2, u_x * Scale1, v_x * Scale1)
    f2 = BB2(u, v, v_t * Scale2, u_x * Scale1, v_x * Scale1, u_xxx*(Scale1**3))

    return f1, f2

def L_bb_Gaussian_PDE_2(input,Model,Scale1,Scale2):
    x, t = input[:, 0], input[:, 1]
    pred = Model(torch.concat((x.unsqueeze(-1), t.unsqueeze(-1)),axis=1))
    u_pred = pred[:, 0]
    v_pred = pred[:, 1]

    u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    u_t = torch.autograd.grad(v_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    v_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    v_t = torch.autograd.grad(v_pred, t, torch.ones_like(u_pred), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    uv_pred = u_pred * v_pred
    uv_x = torch.autograd.grad(uv_pred, x, torch.ones_like(uv_pred), create_graph=True)[0]

    f1 = BB1(u_pred, u_t*Scale2, u_x*Scale1, v_x*Scale1)
    f2 = BB2(u_pred, v_pred, v_t*Scale2, u_x*Scale1, v_x*Scale1, u_xxx*(Scale1)**3)

    return f1 + f2

    

def L_bb_Gaussian_PDE(input,Model,Scale1,Scale2):
    P_0,P_1,P_2 = Model.diff_nm(input) # (input, #mean ,) , (input, #mean,2) , (input, #mean,2,2)
    property_u = Model.property[:, 0].unsqueeze(0)
    property_v = Model.property[:, 1].unsqueeze(0)
    
    u = torch.sum(property_u * P_0, dim=1)
    v = torch.sum(property_v * P_0, dim=1)
    
    u_x = torch.sum(property_u * P_1[:,:,0], dim=1)
    v_x = torch.sum(property_v * P_1[:,:,0], dim=1)
    
    u_t = torch.sum(property_u * P_1[:,:,1], dim=1)
    v_t = torch.sum(property_v * P_1[:,:,1], dim=1)
    
    u_xx = torch.sum(property_u * P_2[:,:,0,0], dim=1)

    u_grad = torch.autograd.grad(u_xx, input, torch.ones(input[:,0].shape).to(device), create_graph=True)[0]
    u_xxx = u_grad[:, 0]
    
    f1 = BB1(u,u_t*Scale2,u_x*Scale1,v_x*Scale1)
    f2 = BB2(u,v,v_t*Scale2,u_x*Scale1,v_x*Scale1,u_xxx*(Scale1)**3)

    return (f1 + f2).unsqueeze(-1)
      
def L_NS_Gaussian_PDE(input,Model,Scale,Re):
    
    P_0,P_1,P_2 = Model.diff_nm(input) # (input, #mean ,) , (input, #mean,2) , (input, #mean,2,2)
    property_u = Model.property[:, 0].unsqueeze(0)
    property_v = Model.property[:, 1].unsqueeze(0)
    property_p = Model.property[:, 2].unsqueeze(0)
    
    u = torch.sum(property_u * P_0, dim=1)
    v = torch.sum(property_v * P_0, dim=1)
    
    u_x = torch.sum(property_u * P_1[:,:,0], dim=1)
    u_y = torch.sum(property_u * P_1[:,:,1], dim=1)
    
    v_x = torch.sum(property_v * P_1[:,:,0], dim=1)
    v_y = torch.sum(property_v * P_1[:,:,1], dim=1)
    
    p_x = torch.sum(property_p * P_1[:,:,0], dim=1)
    p_y = torch.sum(property_p * P_1[:,:,1], dim=1)
    
    u_xx = torch.sum(property_u * P_2[:,:,0,0], dim=1)
    u_yy = torch.sum(property_u * P_2[:,:,1,1], dim=1)
    
    v_xx = torch.sum(property_v * P_2[:,:,0,0], dim=1)
    v_yy = torch.sum(property_v * P_2[:,:,1,1], dim=1)
    
    Eu = u*u_x + v*u_y + p_x - Scale * (1/Re)*(u_xx + u_yy)
    Ev = u*v_x + v*v_y + p_y - Scale * (1/Re)*(v_xx + v_yy)
    Ec = (u_x + v_y)

    
    return Eu.reshape(-1,1),Ev.reshape(-1,1),Ec.reshape(-1,1)

def L_Gaussian_auto(x,t,Model):
    input = torch.concat((x,t),axis=1)
    pred = Model(input)
    import time 
    start = time.time()
    u_x = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(pred.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    f = Burgers(pred,u_x,u_t,u_xx)
    end = time.time()
    
    print(end-start)
    return pred,f

def L_pinn_PDE(x,y,model,data_type,_x_L:float=1.0,_t_L:float=10.0,_Re:float=0.3):
    sol = model(x, y)
    if data_type == 'bur':
        u = sol[:,0].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        f = Burgers(u,u_x,u_t,u_xx)
    
    elif data_type == 'bb':
        u = sol[:,0].reshape(-1,1)
        v = sol[:,1].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
        
        u_t = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        
        f1 = BB1(u,u_t/_t_L,u_x/_x_L,v_x/_x_L)
        f2 = BB2(u,v,v_t/_t_L,u_x/_x_L,v_x/_x_L,u_xxx/(_x_L)**3)
        f = f1 + f2
        
    else:
        u = sol[:,0].reshape(-1,1)
        v = sol[:,1].reshape(-1,1)
        p = sol[:,2].reshape(-1,1)
        
        u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
        
        Eu = u*u_x + v*u_y + p_x - (1/_Re)*(u_xx + u_yy)
        Ev = u*v_x + v*v_y + p_y - (1/_Re)*(v_xx + v_yy)
        Ec = u_x + v_y
        f = Eu + Ev + Ec
    
    return sol,f 

def Burgers(u, u_x, u_y, u_xx): 
    return u_y + u * u_x - (0.01 / np.pi) * u_xx

def BB1(u, u_t, u_x, v_x): 
        return u_t - 2 * u * u_x - 0.5 * v_x

def BB2(u, v, v_t, u_x, v_x, u_xxx): 
    return v_t - 0.5 * u_xxx - 2 * u * v_x - 2 * u_x * v