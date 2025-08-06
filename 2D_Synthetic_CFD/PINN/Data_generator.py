import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import argparse
import scipy.io
import sympy as sy

def load_data_burgers(Data_num=10000,save_fig=False):
    
    data = scipy.io.loadmat('./Data/burgers_shock.mat') # You can download data at https://github.com/maziarraissi/PINNs/tree/master/main/Data
    y = data['t'].flatten() # 100; range [0, 1]
    x = data['x'].flatten() # 256; range [-1, 1]
    u = np.real(data['usol']) # 256, 100
    
    ### Boundary
    xb = np.concatenate([x, np.zeros(len(y))+1, np.zeros(len(y))-1])
    yb = np.concatenate([np.zeros(len(x)), y, y])
    ub = np.concatenate([u[:, 0], u[-1, :], u[0, :]])
    
    y, x = np.meshgrid(y, x)
    x, y, u_s = x.reshape(-1), y.reshape(-1), u.reshape(-1)
    f = np.zeros(len(x))
    
    N_u = 300 # Number of boundary points
    N_f = Data_num # Number of collocation points
    
    idx = np.random.choice(u_s.shape[0], N_f, replace=False)
    
    uf = u_s[idx]
    ff = f[idx]
    xf = x[idx]
    yf = y[idx]
    
    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx]

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.tricontourf(x,y, u.reshape(-1) ,100,cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u(x,t)')
        plt.colorbar()
        plt.show()
        
    
    return x, y, u, xb, yb, ub,xf, yf,uf,ff

def load_data_burgers_Grid_res(res=100):
    
    data = scipy.io.loadmat('./Data/burgers_shock.mat') # You can download data at https://github.com/maziarraissi/PINNs/tree/master/main/Data
    t = data['t'].flatten() # 100; range [0, 1]
    x = data['x'].flatten() # 256; range [-1, 1]
    
    u = np.real(data['usol']) # 256; range [-1, 1]
    
    idx_x = np.linspace(0, x.shape[0] - 1, res, dtype=int)
    idx_t = np.linspace(0, t.shape[0] - 1, res, dtype=int)
    # idx_x = np.linspace(0, x.shape[0] - 1, len(x)//2, dtype=int)
    # idx_t = np.linspace(0, t.shape[0] - 1, len(t)//2, dtype=int)
    
    u = np.real(data['usol']) # 256, 100
    
    xb = np.concatenate([x, np.zeros(len(t))+1, np.zeros(len(t))-1])
    tb = np.concatenate([np.zeros(len(x)), t, t])
    ub = np.concatenate([u[:, 0], u[-1, :], u[0, :]])
    
    # x_test, t_test = np.meshgrid(x, t)
    # x_test, t_test = x_test.reshape(-1), t_test.reshape(-1)
    
    t_test, x_test = np.meshgrid(t, x)
    x_test, t_test = x_test.reshape(-1), t_test.reshape(-1)
    
    t_train,x_train = np.meshgrid(t[idx_t],x[idx_x])
    x_train, t_train = x_train.reshape(-1), t_train.reshape(-1)
    u_train = u[np.ix_(idx_x, idx_t)]
    u_train = u_train.reshape(-1)
    f = np.zeros_like(u_train)
    
    u_test = u.reshape(-1)
    
    return x_test, t_test, u_test, xb, tb, ub,  x_train,t_train,u_train,f


def load_data_burgers_random(Data_num=10000,save_fig=False):
    
    data = scipy.io.loadmat('./Data/burgers_shock.mat') # You can download data at https://github.com/maziarraissi/PINNs/tree/master/main/Data
    y = data['t'].flatten() # 100; range [0, 1]
    x = data['x'].flatten() # 256; range [-1, 1]
    u = np.real(data['usol']) # 256, 100
    
    ### Boundary
    xb = np.concatenate([x, np.zeros(len(y))+1, np.zeros(len(y))-1])
    yb = np.concatenate([np.zeros(len(x)), y, y])
    ub = np.concatenate([u[:, 0], u[-1, :], u[0, :]])
    
    y, x = np.meshgrid(y, x)
    x, y, u_s = x.reshape(-1), y.reshape(-1), u.reshape(-1)
    f = np.zeros(len(x))
    
    N_u = 300 # Number of boundary points
    N_f = Data_num # Number of collocation points
    
    idx = np.random.choice(u_s.shape[0], N_f, replace=False)
    
    uf = u_s[idx]
    ff = f[idx]
    xf = x[idx]
    yf = y[idx]
    
    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]; xb = xb[idx]; yb = yb[idx]

    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.tricontourf(x,y, u.reshape(-1) ,100,cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u(x,t)')
        plt.colorbar()
        plt.show()
        
    
    return x, y, u, xb, yb, ub,xf, yf,uf,ff

def load_data_BB_Grid_res(res:int=100,x0=-10, x1=15, t0=-3, t1=2, save_fig=0):
    
    p = 1; q = 1; beta = 1; p1 = 2; q1 = p1*(2*q+p1*p+2*p**2)/(2*p)
    x, t = sy.symbols('x, t'); w = p * x + q * t + 0.5 * sy.log(1 + sy.exp(p1 * x + q1 * t))
    u1 = w.diff(x) / 2
    u0 = (2 * w.diff(t) - w.diff(x).diff(x)) / (4 * w.diff(x))
    v2 = (beta / 2 - 1) * w.diff(x) * w.diff(x)
    v1 = (1 - beta / 2) * w.diff(x).diff(x)
    v0 = - (beta - 2) * \
        (2 * (w.diff(x))**4 - w.diff(x) * w.diff(x).diff(x).diff(x) + 2 * w.diff(x) * w.diff(x).diff(t) + (w.diff(x).diff(x))**2 - 2 * w.diff(x).diff(x) * w.diff(t)) / \
            (4 * w.diff(x) * w.diff(x))
    u = u0 + u1 * sy.tanh(w)
    v = v0 + v1 * sy.tanh(w) + v2 * (sy.tanh(w))**2
    func_u = sy.lambdify([x, t], u,'numpy')
    func_v = sy.lambdify([x, t], v,'numpy')
    
    x_L = x1 - x0
    t_L = t1 - t0
    x, t = np.linspace(x0, x1, res), np.linspace(t0, t1, res); x, t = np.meshgrid(x, t); x, t = x.reshape(-1), t.reshape(-1);
    u, v = func_u(x, t), func_v(x, t)

    xb = np.concatenate([np.linspace(x0, x1, res), np.linspace(x0, x1, res), np.zeros(res) + x0, np.zeros(res) + x1], axis=0)
    tb = np.concatenate([np.zeros(res) + t0, np.zeros(res) + t1, np.linspace(t0, t1, res), np.linspace(t0, t1, res)], axis=0)
    ub, vb = func_u(xb, tb), func_v(xb, tb)

    
    return x, t, u, v, xb, tb, ub, vb,x_L,t_L

def load_data_BB_random(Num,x0=-10, x1=15, t0=-3, t1=2, save_fig=0):
    p = 1; q = 1; beta = 1; p1 = 2; q1 = p1*(2*q+p1*p+2*p**2)/(2*p)
    x, t = sy.symbols('x, t'); w = p * x + q * t + 0.5 * sy.log(1 + sy.exp(p1 * x + q1 * t))
    u1 = w.diff(x) / 2
    u0 = (2 * w.diff(t) - w.diff(x).diff(x)) / (4 * w.diff(x))
    v2 = (beta / 2 - 1) * w.diff(x) * w.diff(x)
    v1 = (1 - beta / 2) * w.diff(x).diff(x)
    v0 = - (beta - 2) * \
        (2 * (w.diff(x))**4 - w.diff(x) * w.diff(x).diff(x).diff(x) + 2 * w.diff(x) * w.diff(x).diff(t) + (w.diff(x).diff(x))**2 - 2 * w.diff(x).diff(x) * w.diff(t)) / \
            (4 * w.diff(x) * w.diff(x))
    u = u0 + u1 * sy.tanh(w)
    v = v0 + v1 * sy.tanh(w) + v2 * (sy.tanh(w))**2
    func_u = sy.lambdify([x, t], u,'numpy')
    func_v = sy.lambdify([x, t], v,'numpy')
    
    x_L = x1 - x0
    t_L = t1 - t0
    x, t = np.linspace(x0, x1, 201), np.linspace(t0, t1, 201); x, t = np.meshgrid(x, t); x, t = x.reshape(-1), t.reshape(-1);
    u, v = func_u(x, t), func_v(x, t)

    xb = np.concatenate([np.linspace(x0, x1, 201), np.linspace(x0, x1, 201), np.zeros(201) + x0, np.zeros(201) + x1], axis=0)
    tb = np.concatenate([np.zeros(201) + t0, np.zeros(201) + t1, np.linspace(t0, t1, 201), np.linspace(t0, t1, 201)], axis=0)
    ub, vb = func_u(xb, tb), func_v(xb, tb)

    N_u = 400 # Number of boundary points
    N_f = Num # Number of collocation points

    idx = np.random.choice(u.shape[0], N_f, replace=False)
    xf = x[idx]
    tf = t[idx]
    uf = u[idx]
    vf = v[idx]
    
    # Boundary PINN selection
    idx = np.random.choice(ub.shape[0], N_u, replace=False)
    ub = ub[idx]
    vb = vb[idx]
    xb = xb[idx]
    tb = tb[idx]

    
    if save_fig:
        plt.rcParams["figure.figsize"] = (6.0,5.0)
        plt.scatter(xf,tf,s=3)
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(x, t)
        plt.tricontourf(triang_coarse, u, 100 ,cmap='jet', extend="both")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers u(x,t)')
        plt.colorbar()
        plt.show()

        plt.rcParams["figure.figsize"] = (6.0,5.0)
        triang_coarse = tri.Triangulation(x, t)
        plt.tricontourf(triang_coarse, v, 100 ,cmap='jet', extend="both")
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Burgers v(x,t)')
        plt.colorbar()
        plt.show()
        
    return x, t, u, v, xb, tb, ub, vb, xf, tf,uf, vf ,x_L,t_L
