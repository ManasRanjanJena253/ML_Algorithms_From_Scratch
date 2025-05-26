# Importing dependencies
from collections import Counter
import numpy as np

# KNN Algorithm working :
# 1. Chooses K : K is the nearest neighbours that you want your model to consider. A low value of K have risk of overfitting and a low value have a risk of underfitting.
# 2. Compute distance : Uses a distance metric to find the relation between various points. Usually uses euclidean distance.
# 3. Select K nearest neighbours : Sort by distance and pick the top K.
# 4. Vote : Count the class labels of the K neighbours
# 5. Assign Class : Assign the class with the highest vote to the query point

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN :
    def __init__(self, k) :
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, X):
        # Computing the distances
        distances = [euclidean_distance(X, X_train) for X_train in self.X_train]

        # Getting the k-nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Getting the majority votes/ most common class labels
        most_common = Counter(k_nearest_labels).most_common(1)  # The most common method gives a list containing tuples of the element and its no. of occurrences.
        return most_common[0][0]