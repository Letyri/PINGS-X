import torch


def find_closest_distances(points, scale: float = 1.0):
    num_points = points.shape[0]
    distances = torch.cdist(points, points, p=3)  # (num_points, num_points)
    if torch.any(torch.isnan(distances)):
        raise ValueError("Distances tensor contains NaN values after cdist")
    
    distances += torch.eye(num_points, device=points.device) * float(2500)
    min_distances, _ = torch.min(distances, dim=1)  # (num_points,)
    
    return min_distances * scale

def find_closest_distances_tmp(points, scale: float = 1.0):
    # Reshape the 4D tensor to a 2D tensor (num_points, 3)
    num_points = points.shape[0] * points.shape[1] * points.shape[2]
    points_reshaped = points.view(num_points, 3)
    
    # Calculate pairwise distances
    distances = torch.cdist(points_reshaped, points_reshaped, p=2)  # (num_points, num_points)
    if torch.any(torch.isnan(distances)):
        raise ValueError("Distances tensor contains NaN values after cdist")
    
    # Add a large value to the diagonal to ignore self-distances
    distances += torch.eye(num_points, device=points.device) * 2500  # (num_points, num_points)
    
    # Find the minimum distances
    min_distances, _ = torch.min(distances, dim=1)  # (num_points,)
    
    return min_distances * scale

def Cal_q(quaternions):
    assert quaternions.shape[-1] == 4, "Input quaternions must have shape (N, 4)"
    
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    w, x, y, z = quaternions.unbind(dim=-1)
    
    R = torch.stack([
        torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)], dim=-1),
        torch.stack([2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)], dim=-1),
        torch.stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)], dim=-1)
    ], dim=-2)  
    
    R_T = torch.stack([
        torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - w*-z), 2 * (x*z + w*-y)], dim=-1),
        torch.stack([2 * (x*y + w*-z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*-x)], dim=-1),
        torch.stack([2 * (x*z - w*-y), 2 * (y*z + w*-x), 1 - 2 * (x**2 + y**2)], dim=-1)
    ], dim=-2)
    
    return R, R_T
    


