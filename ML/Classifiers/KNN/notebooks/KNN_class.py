import numpy as np
from collections import Counter

def euclidean_distance(P, Q):
    """Function returning row-wise Euclidean Distance values for 
    matrices P and Q. Works with just vectors.
    """
    return np.linalg.norm(P - Q, axis=1)

def manhattan_distance(P, Q):
    """Function returning row-wise Manhattan Distance values for 
    matrices P and Q. Works with just vectors.
    """
    return np.abs(P - Q).sum(axis=1)

class KNN:
    """K Nearest Neighbor algorithm implementation"""
    
    def __init__(self, k, distance_func):
        """Constructor of KNN function.
        
        Parameters:
        -----------
        k: int
            Number of neighbors that will be used in class vote.
        distance_func: function
            Function for calculating distance between samples.
        """
        
        self.k = k
        self.distance_func = distance_func
        
    def _calculate_distances(self, X_rep, sample):
        """Calculate distance according to distance_func between single sample and all samples from
        X_rep matrix.
        
        Parameters:
        -----------
        sample: numpy.ndarray
            Vector containing feature values of single sample.
        X_rep: numpy.ndarray
            Matrix of samples based on which predictions will be made.
        
        Returns:
        -----------
        distances: numpy.ndarray
            Returns 1-dimensional ndarray containing calculated distances.
        """
        distances = self.distance_func(X_rep, sample)
        return distances
    
    def _get_k_neighbors(self, distances):
        """Returns indices of k lowest numbers in distances ndarray.
        
        Parameters:
        -----------
        distances: numpy.ndarray
            1-dimensional ndarray containing distances between single sample and all samples in 
            X_rep matrix.
        
        Returns:
        -----------
        top_k_neighbours: numpy.ndarray
            Returns 1-dimensional ndarray with indices of lowest k numbers in distances ndarray.
        """
        top_k_neighbours = np.argpartition(distances, self.k)[:self.k]
        return top_k_neighbours
    
    def _vote(self, neighbors, y_rep):
        """Returns most frequent class id in neighbour group.
        
        Parameters:
        -----------
        neighbors: numpy.ndarray
            1-dimensional ndarray with indices of lowest k numbers in distances ndarray.
        y_rep: numpy.ndarray
            Vector containing class ids for each row in matrix X_rep.
        
        Returns:
        -----------
        most_frequent_class: int
            Returns most frequent class id.
        """
        most_frequent_class = np.bincount(y_rep[neighbors]).argmax()
        return most_frequent_class
        
    def _predict_for_sample(self, sample, X_rep, y_rep):
        """Implementation of KNN for single sample:
        1. Calculates distance betwen sample and all rows of X_rep.
        2. Sorts calculated distances and picks ids of rows which distance 
           is closest to sample.
        3. Selects subset of k labels from vector y_rep based on previously
           picked indices.
        4. Returns most appearing class.
        
        Parameters:
        -----------
        sample: numpy.ndarray
            Vector containing feature values of single sample.
        X_rep: numpy.ndarray
            Matrix of samples based on which predictions will be made.
        y_rep: numpy.ndarray
            Vector containing class ids for each row in matrix X_rep.
            
        Returns:
        -----------
        predicted_class: int
            Returns class id of most similar class to given sample.
        """
        distances = self._calculate_distances(X_rep, sample)
        k_neighbors = self._get_k_neighbors(distances)
        predicted_class = self._vote(k_neighbors, y_rep)
        return predicted_class
            
    def predict(self, X_new, X_rep, y_rep):
        """Function that for each row of matrix X_new performs classification 
        based on matrix X_rep and it's labels y_rep.
        
        Parameters:
        -----------
        X_new: numpy.ndarray
            Matrix of samples for which predictions will be made.
        X_rep: numpy.ndarray
            Matrix of samples based on which predictions will be made.
        y_rep: numpy.ndarray
            Vector containing class ids for each row in matrix X_rep.
            
        Returns:
        -----------
        result: numpy.ndarray
            Vector containing predicted class ids for each row in 
            matrix X_new.
        """
        return np.array([self._predict_for_sample(s, X_rep, y_rep) for s in X_new])