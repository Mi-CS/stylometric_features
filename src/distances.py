
import numpy as np
from scipy.spatial.distance import cdist

def mod_hausdorff_dist(pc1, pc2): 
    """
    Compute the modified Hausdorff distance between
    two point clouds, as defined by: 
    D(A, B) = max(d(A, B), d(B, A))
    d(A, B) = 1/N_a  Σ_a d(a, B)
    d(a, B) = min_b ||a-b||

    Parameters: 
        pc1 (numpy.ndarray): Numpy ndarray of dimensions
            n_points x space_dimensions
        pc2 (numpy.ndarray): Numpy ndarray of dimensions
            n_points x space_dimensions
    
    Return: 
        mh_distance (float): Modified Hausdorff distance
            between pc1 and pc2
    """
    
    distances = cdist(pc1, pc2)
    d_aB = distances.min(axis=1)
    d_bA = distances.min(axis=0)
    
    dAB = sum(d_aB) / len(d_aB)
    dBA = sum(d_bA) / len(d_bA)
    
    return max(dAB, dBA)