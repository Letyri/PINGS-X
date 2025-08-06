import torch

def find_closest_distances(points):
    num_points = points.shape[0]
    distances = torch.cdist(points, points, p=2)  # (num_points, num_points)
    if torch.any(torch.isnan(distances)):
        raise ValueError("Distances tensor contains NaN values after cdist")
    
    distances += torch.eye(num_points, device=points.device) * float(2500)
    min_distances, _ = torch.min(distances, dim=1)  # (num_points,)
    
    return min_distances # / 2.0

def find_closest_distances_PDE(points):
    num_points = points.shape[0]
    distances = torch.cdist(points, points, p=2)  # (num_points, num_points)
    if torch.any(torch.isnan(distances)):
        raise ValueError("Distances tensor contains NaN values after cdist")
    
    distances += torch.eye(num_points, device=points.device) * float(2500)
    min_distances, _ = torch.min(distances, dim=1)  # (num_points,)
    
    return min_distances * 10.0

def rotation_cal(rotation):
    rotation = rotation % torch.pi
    Rot_cos = torch.cos(rotation)
    Rot_sin = torch.sin(rotation)
    
    Matrix_rotation1 = torch.zeros((rotation.shape[0], 2),device=rotation.device)
    
    Matrix_rotation1[:, 0] = Rot_cos
    Matrix_rotation1[:, 1] = Rot_sin
    
    return Matrix_rotation1

def Cal_rot(rotation):
    rotation = rotation % torch.pi
    Rot_cos = torch.cos(rotation)
    Rot_sin = torch.sin(rotation)
    
    Matrix_rotation1 = torch.zeros((rotation.shape[0], 2, 2),device=rotation.device)
    Matrix_rotation2 = torch.zeros((rotation.shape[0], 2, 2),device=rotation.device)
    
    Matrix_rotation1[:, 0, 0] = Rot_cos
    Matrix_rotation1[:, 0, 1] = -Rot_sin
    Matrix_rotation1[:, 1, 0] = Rot_sin
    Matrix_rotation1[:, 1, 1] = Rot_cos
    
    Matrix_rotation2[:, 0, 0] = Rot_cos
    Matrix_rotation2[:, 0, 1] = Rot_sin
    Matrix_rotation2[:, 1, 0] = -Rot_sin
    Matrix_rotation2[:, 1, 1] = Rot_cos
    
    
    return Matrix_rotation1,Matrix_rotation2

def Cal_Scale(scale):
    scale_rotation = torch.zeros((scale.shape[0], 2, 2),device=scale.device)
    
    scale_rotation[:, 0, 0] = -scale[:,0].exp()
    scale_rotation[:, 1, 1] = -scale[:,1].exp()
    
    return scale_rotation

