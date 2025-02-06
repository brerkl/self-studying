import numpy as np
from collections import Counter

def manhattan_distance(x1, x2):
    distance = np.sum(np.absolute(x1-x2))
    return distance

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def minkowski_distance(x1, x2, p):
    if p > 0 & isinstance(p, int):
        distance = (np.sum(np.abs(x1-x2))**p) ** (1/p)
        return distance
    return "Biến \"p\" phải là một số nguyên."

class KNearestClassifier:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # compute distance
        distances = [manhattan_distance(x, x_train) if self.p == 1 else 
                    euclidean_distance(x, x_train) if self.p == 2 else 
                    minkowski_distance(x, x_train, self.p) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # majority voye
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common