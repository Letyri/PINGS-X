import numpy as np
import torch
import networkx as nx 
import random
import math 
import time 

from torch.autograd import Variable

device = 'cuda'

def make_clusters(edge_tensors):
    edges = [(edge[0].item(), edge[1].item()) for edge in edge_tensors]
    G = nx.Graph()
    G.add_edges_from(edges)
    connected_components = list(nx.connected_components(G))
    clusters = [sorted(list(component)) for component in connected_components]

    return clusters
    
def compute_cosine_similarity_matrix(data):
    # 25600 x 1600
    features = data.T  # 1600 x 25600
    norms = torch.norm(features, dim=1, keepdim=True)
    
    normalized_features = features / norms
    
    similarity_matrix = torch.mm(normalized_features, normalized_features.T)
    
    return similarity_matrix 

def dpdn(p_x, p_y, n):
    p_n = p_x*n[:,0].reshape(-1,1) + p_y*n[:,1].reshape(-1,1)
    p_n.reshape(-1,1)
    return p_n

def relative_l2(u, u_gt):
    return torch.norm(u - u_gt) / torch.norm(u_gt)

def MSELoss(pred,target):
    Loss = torch.sum((pred - target) ** 2, dim=1)
    
    return Loss 