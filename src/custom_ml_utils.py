
"""
Custom implementations of common machine learning utilities
"""

import numpy as np
from collections import Counter
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TrainTestSplit:
    """
    A custom implementation of train_test_split that mimics sklearn's behavior
    with support for stratified sampling and random state for reproducibility.
    """
    
    def __init__(self):
        self.random_state = None
        self.rng = None
    
    def __call__(self, X, y, test_size=0.2, random_state=None, stratify=None):
        """
        Split arrays or matrices into random train and test subsets.
        
        Parameters:
        -----------
        X : array-like
            The input samples to split.
        y : array-like
            The target values for supervised learning problems.
        test_size : float, default=0.2
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        random_state : int, default=None
            Controls the randomness of the training and testing indices.
        stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels.
            
        Returns:
        --------
        X_train : array-like
            The training input samples.
        X_test : array-like
            The testing input samples.
        y_train : array-like
            The training target values.
        y_test : array-like
            The testing target values.
        """
        # Set random state for reproducibility
        if random_state is not None:
            self.random_state = random_state
            self.rng = np.random.RandomState(random_state)
        else:
            if self.rng is None:
                self.rng = np.random.RandomState()
        
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get number of samples
        n_samples = X.shape[0]
        
        # Calculate test size
        if isinstance(test_size, float):
            n_test = int(n_samples * test_size)
        else:
            n_test = test_size
            
        n_train = n_samples - n_test
        
        # If stratify is None, do a simple random split
        if stratify is None:
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            return X_train, X_test, y_train, y_test
        
        # If stratify is provided, do stratified sampling
        stratify = np.asarray(stratify)
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        
        # Initialize arrays for indices
        train_indices = []
        test_indices = []
        
        # For each class, sample proportionally
        for cls, count in zip(unique_classes, class_counts):
            # Get indices of samples belonging to this class
            cls_indices = np.where(stratify == cls)[0]
            
            # Calculate number of test samples for this class
            n_test_cls = int(count * test_size)
            n_train_cls = count - n_test_cls
            
            # Shuffle the indices for this class
            self.rng.shuffle(cls_indices)
            
            # Split into train and test
            train_indices.extend(cls_indices[:n_train_cls])
            test_indices.extend(cls_indices[n_train_cls:])
        
        # Convert to numpy arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Shuffle the final indices to mix classes
        self.rng.shuffle(train_indices)
        self.rng.shuffle(test_indices)
        
        # Split the data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test


class KMeans:
    """
    Custom implementation of K-Means clustering algorithm
    """
    
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, 
                 tol=1e-4, random_state=None):
        """
        Initialize K-Means clustering
        
        Parameters:
        -----------
        n_clusters : int, default=8
            The number of clusters to form as well as the number of centroids to generate.
        init : {'k-means++', 'random'}, default='k-means++'
            Method for initialization.
        n_init : int, default=10
            Number of time the k-means algorithm will be run with different centroid seeds.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float, default=1e-4
            Relative tolerance with regards to inertia to declare convergence.
        random_state : int, default=None
            Determines random number generation for centroid initialization.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """Initialize centroids using k-means++ algorithm"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        # Choose first centroid randomly
        first_idx = np.random.choice(n_samples)
        centroids[0] = X[first_idx]
        
        # Choose remaining centroids
        for i in range(1, self.n_clusters):
            # Calculate distances to existing centroids
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist = float('inf')
                for k in range(i):
                    dist = np.sum((X[j] - centroids[k])**2)
                    if dist < min_dist:
                        min_dist = dist
                distances[j] = min_dist
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / np.sum(distances)
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids[i] = X[next_idx]
            
        return centroids
    
    def _random_init(self, X):
        """Random initialization of centroids"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]
    
    def _compute_distances(self, X, centroids):
        """Compute distances from each point to each centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        return distances
    
    def _assign_clusters(self, distances):
        """Assign each point to the nearest centroid"""
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, X, labels):
        """Compute new centroids as mean of points in each cluster"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep the old centroid
                new_centroids[i] = self.cluster_centers_[i]
        return new_centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """Compute the sum of squared distances of samples to their closest cluster center"""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i])**2)
        return inertia
    
    def fit(self, X):
        """Compute k-means clustering"""
        X = np.asarray(X)
        best_inertia = float('inf')
        best_labels = None
        best_centers = None
        best_n_iter = 0
        
        for _ in range(self.n_init):
            # Initialize centroids
            if self.init == 'k-means++':
                centroids = self._initialize_centroids(X)
            else:  # random
                centroids = self._random_init(X)
            
            # Run k-means algorithm
            for iteration in range(self.max_iter):
                # Compute distances and assign clusters
                distances = self._compute_distances(X, centroids)
                labels = self._assign_clusters(distances)
                
                # Compute new centroids
                new_centroids = self._compute_centroids(X, labels)
                
                # Check for convergence
                centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - centroids)**2, axis=1)))
                centroids = new_centroids
                
                if centroid_shift <= self.tol:
                    break
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)
            
            # Keep the best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centroids
                best_n_iter = iteration + 1
        
        # Store the best result
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample"""
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to"""
        X = np.asarray(X)
        distances = self._compute_distances(X, self.cluster_centers_)
        return self._assign_clusters(distances)
