import numpy as np
from typing import Literal
from collections import Counter

from distances import mod_hausdorff_dist

def get_frag_dist(pc_list: list) -> np.ndarray:
    """
    Take a list of point clouds and compute
    the Modified Hausdorff distance between 
    them. Each entry of the list represents
    a document and contains the point clouds
    of the fragments for that document. 

    By default, the distance between any two 
    fragments from the same text is -1. 
    """ 
    # Create a flatten list of indices indicating
    # to which document each fragment belongs to
    doc_lengths = [len(doc) for doc in pc_list]
    doc_idx = [[i]*length for i, length in enumerate(doc_lengths)]
    doc_idx = [x for y in doc_idx for x in y]

    # Flatten point clouds
    pc_list = [pc for doc in pc_list for pc in doc]

    # Compute distance matrix
    dist_matrix = np.zeros(shape=(len(pc_list), len(pc_list)))
    for i, frag_i in enumerate(pc_list):
        for j, frag_j in enumerate(pc_list): 
            if i <= j:
                continue
            if doc_idx[i] == doc_idx[j]:
                dist_matrix[i, j] = -1
                continue
            dist_matrix[i, j] = mod_hausdorff_dist(frag_i, frag_j)

    dist_matrix += dist_matrix.T
    np.fill_diagonal(dist_matrix, -1)

    return dist_matrix


def get_adjacency(dist_matrix: np.ndarray,
                  doc_idx: np.ndarray,
                  method: Literal["closest", "threshold"],
                  n_fragments: int = 4,
                  threshold: float = 0.1, 
                  min_fragments: int = 1):

    """
    Compute the document adjacency matrix according to the specified method from the
    fragment distance matrix. If 'method=closest', for each fragment keeps the 
    closest 'n_fragments'. If 'method=threshold', distances are normalized to 0-1, and
    keep all fragments below 'threshold'. Fragments belonging to the same document are
    considered to be infinitely separated. It then links each pair of documents if at 
    least 'min_fragments' are linked.    

    Parameters: 
        dist_matrix (numpy.ndarray): Distance matrix of fragments
        doc_idx (numpy.ndarray): Array of length number of fragments. Each entry 
            indicates to which document each fragment belongs to. 
        method (str) ('closest' or 'threshold'): Method employed to compute the 
            document adjacency matrix
        n_fragments (int) (Optional, def: 4): Only used when 'method=closest'. Number of 
            closest fragments to each single fragments to take into account. 
        threshold (float) (Optional, def: 0.1): Only used when 'method=threshold'. 
            Maximum distance to link fragments. 
        min_fragments (int) (Optional, def: 1): Minimum number of linked fragments
            needed to link two documents. By default 1, i.e., consider all linked 
            fragments. 

    Returns: 
        adjacency_matrix (numpy.ndarray): Adjacency matrix of documents
    """

    dist_matrix = np.where(dist_matrix == -1, np.inf, dist_matrix)

    assert method in ["closest", "threshold"], "method can only be 'closest' or 'threshold'"

    if method == "closest": 
        return _get_adjacency_closest(dist_matrix, doc_idx, n_fragments)
    if method == "threshold": 
        return _get_adjacency_threshold(dist_matrix, doc_idx, threshold)

def _get_adjacency_closest(dist_matrix, doc_idx, n_fragments): 
    closest_frag_idx = dist_matrix.argsort(axis=1)[:, :n_fragments]

    def get_doc_idx(frag_idx):
        return doc_idx[frag_idx]
    get_doc_idx = np.vectorize(get_doc_idx)

    closest_doc_idx = get_doc_idx(closest_frag_idx)
    
    return _build_adjacency_matrix_closest(closest_doc_idx, doc_idx)

def _get_adjacency_threshold(dist_matrix, doc_idx, threshold):

    # Normalize dist_matrix
    min_dist = dist_matrix.min()
    max_dist = dist_matrix[dist_matrix != np.inf].max()
    dist_matrix = (dist_matrix - min_dist) / (max_dist - min_dist + 1e-30)

    # Build matrix with doc indices
    idx_matrix = np.ones_like(dist_matrix)*doc_idx

    # Mask to -1 according to threshold
    idx_matrix = np.where(dist_matrix > threshold, -1, idx_matrix).astype(int)
    # Get indices where to cut
    counts = Counter(doc_idx)
    cut_idx = np.cumsum([counts[i] for i in range(len(counts))])[:-1]

    # Build adjacency matrix
    closest_docs = np.split(idx_matrix, cut_idx)
    adj_matrix = np.zeros((len(closest_docs), len(closest_docs)))
    for i, doc in enumerate(closest_docs): 
        closest_docs_idx = Counter(doc.flatten())

        for idx in closest_docs_idx: 
            if idx == -1:
                continue
            adj_matrix[i, idx] += closest_docs_idx[idx]

    return adj_matrix



def _build_adjacency_matrix_closest(closest_doc_idx, doc_idx): 
    # Get indices where to cut
    counts = Counter(doc_idx)
    cut_idx = np.cumsum([counts[i] for i in range(len(counts))])[:-1]
    
    # Build adjacency matrix
    closest_docs = np.split(closest_doc_idx, cut_idx)
    adj_matrix = np.zeros((len(closest_docs), len(closest_docs)))
    for i, doc in enumerate(closest_docs): 
        closest_docs_idx = Counter(doc.flatten())        

        for idx in closest_docs_idx: 
            adj_matrix[i, idx] += closest_docs_idx[idx]
    
    return adj_matrix

    
