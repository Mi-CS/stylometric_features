import numpy as np
import re
import scipy.stats as stat
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from typing import Optional, List

from stylographic_features import get_stylographic_feat

from sklearn.decomposition import PCA


def style_point_cloud(text: str, 
                        window_size: int = 600,
                        window_overlap: int = 0,
                        max_tokens: Optional[int] = None,
                        reduce: Optional[str] = None, 
                        redu_kwargs: dict = {}) -> np.ndarray: 
    """
    Takes a text and returns a point cloud where
    each point is a style vector of a text fragment.

    Parameters: 
        text (str): Body of text to be converted to a 
            style point cloud
        window_size (int) (def. 600): Window size of each fragment from
            which the style vector is computed
        window_overlap (int) (def. 0): Token overlap of windows. 
        max_tokens (int) (def. None): Maximum number of tokens to consider 
            from the text to build the point cloud. Default to None, i.e.,
            take all.
        reduce (str): Whether to apply any dimensional reduction
            technique. By default is None and returns a 43-dimensional
            point cloud. Optional values are {"pca"}. 
        redu_kwargs (dict): keyword arguments for sklearn reduction function.
    
    Returns:
        style_vectors (np.ndarray): stacked style vectors for each 
            point in the style point cloud.
    """

    tokenized_text = word_tokenize(text)[:max_tokens] if max_tokens else\
        word_tokenize(text)
    
    # Compute step 
    step = window_size - window_overlap

    # Generator for token chunks
    token_chunks = (tokenized_text[i:i+window_size] for i in 
                        range(0, len(tokenized_text) - window_size, 
                                step))

    # Compute style vectors
    style_vectors = np.empty(shape=(0, 43))
    for chunk in token_chunks: 
        style_vectors = np.vstack((style_vectors,
                                   get_stylographic_feat(chunk)))
    

    if reduce: 
        pca = PCA(**redu_kwargs)
        style_vectors = pca.fit_transform(style_vectors)

    return style_vectors



