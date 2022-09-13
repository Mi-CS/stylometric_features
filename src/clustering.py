from operator import mod
import numpy as np
from itertools import permutations
from distances import mod_hausdorff_dist

from sklearn.cluster import KMeans, AgglomerativeClustering

class KMeansAuthors:
    """
    Uses KMeans to predict the authors of style vectors. 
    """ 

    def __init__(self, n_authors: int) -> None: 
        self.kmeans = KMeans(n_clusters=n_authors)
        self.auth_dict = None
        self.best_score = 0

    def fit(self, X: np.ndarray, author_labels: np.ndarray) -> None: 
        """
        Compute KMeans clustering and identify each cluster with an author
        using the author_labels passed. This is done exhaustively by computing
        the score of each possible combination.

        Parameters:
            X (numpy.ndarray): style vectors
            author_labels (nump.ndarray): author to which each style vector
                belongs to
        """

        # Fit KMeans
        self.kmeans.fit(X)
        # Identify clusters with authors
        predictions = self.kmeans.predict(X)
        self.identify_authors(predictions, author_labels)
    
    def identify_authors(self, predictions: np.ndarray, author_labels: np.ndarray) -> None:
        """
        Identify each cluster with one author. It keeps the labels
        that produce the higher score.
        """
        authors = list(set(author_labels))
        for permutation in permutations(set(predictions)): 
            curr_dic = dict(zip(authors, permutation))
            curr_auth_labels = np.array([curr_dic[auth] for auth in author_labels])
            score = (predictions == curr_auth_labels).sum() / len(predictions)
            if score > self.best_score: 
                self.best_score = score
                self.auth_dict = dict(zip(permutation, authors)) 
    
    def predict(self, X: np.ndarray, author_labels: bool = True) -> np.ndarray:
        """
        Predict the labels for each style vector.

        Parameters:
            X (numpy.ndarray): style vectors
            author_labels (bool) (def. True): whether to return the
                labels as strings with the name of authors or ints.

        Returns: 
            predictions (numpy.ndarray): array containing the 
                predicted labels. 
        """
        predictions = self.kmeans.predict(X)

        if author_labels: 
            predictions = np.array([self.auth_dict[pr] for pr in predictions])
        
        return predictions

    def predict_document(self, X: np.ndarray, doc_lengths: np.ndarray) -> np.ndarray: 
        """
        Predict the author of each document. 

        Parameters: 
            X (numpy.ndarray): Ordered style vectors stacked
            doc_lengths (numpy.ndarray): Ordered length of each document
                indicating which style vector belongs to which document.

        Returns: 
            doc_label (numpy.ndarray): Array containing the author assigned
                to each document. 
        """

        predictions = self.kmeans.predict(X)
        predictions = np.array([self.auth_dict[pr] for pr in predictions])
        cut_idx = np.cumsum(doc_lengths)[:-1]
        doc_labels = np.split(predictions, cut_idx)

        doc_predictions = []
        for doc_label in doc_labels: 
            authors, counts = np.unique(doc_label, return_counts=True)
            doc_predictions.append(authors[np.argmax(counts)])
        
        return np.array(doc_predictions)


class ModHausdorffDocument: 
    """
    Compute the Modified Hausdorff distance between 
    every point cloud and cluster them using the 
    specified method.
    """

    def __init__(self, n_authors: int) -> None:
        self.method = AgglomerativeClustering(n_clusters=n_authors, affinity="precomputed", 
                                                linkage="complete")
        self.auth_dict = None
        self.best_score = 0

    def fit(self, X: np.ndarray, doc_lengths: np.ndarray, author_labels: np.ndarray) -> None: 

        # Compute distance matrix and fit 
        distance_matrix = self.compute_distance_matrix(X, doc_lengths)
        self.method.fit(distance_matrix)

        # Identify clusters with authors
        predictions = self.method.labels_
        self.identify_authors(predictions, author_labels)

    
    def compute_distance_matrix(self, X, doc_lengths):
        cut_idx = np.cumsum(doc_lengths)[:-1]
        doc_point_clouds = np.split(X, cut_idx)

        dist_matrix = np.zeros(shape=(len(doc_point_clouds), len(doc_point_clouds)))
        for i, doc_i in enumerate(doc_point_clouds):
            for j, doc_j in enumerate(doc_point_clouds): 
                if i <= j: 
                    continue
                dist_matrix[i, j] = mod_hausdorff_dist(doc_i, doc_j)

        dist_matrix += dist_matrix.T
        return dist_matrix

    def identify_authors(self, predictions: np.ndarray, author_labels: np.ndarray) -> None:
        """
        Identify each cluster with one author. It keeps the labels
        that produce the higher score.
        """
        authors = list(set(author_labels))
        for permutation in permutations(set(predictions)): 
            curr_dic = dict(zip(authors, permutation))
            curr_auth_labels = np.array([curr_dic[auth] for auth in author_labels])
            score = (predictions == curr_auth_labels).sum() / len(predictions)
            if score > self.best_score: 
                self.best_score = score
                self.auth_dict = dict(zip(permutation, authors)) 

    def predict_document(self): 
        return np.array([self.auth_dict[lbl] for lbl in self.method.labels_])







            
