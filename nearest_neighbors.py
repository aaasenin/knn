import numpy as np
import distances
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:

    def __init__(self, k=3, strategy='my_own',
                 metric='euclidean', weights=False,
                 test_block_size=100):
        
        if type(k) != int:
            raise TypeError('wrong k type')
        if strategy != 'my_own' and strategy != 'brute' \
        and strategy != 'kd_tree' and strategy != 'ball_tree':
            raise TypeError('wrong strategy')
        if metric != 'euclidean' and metric != 'cosine':
            raise TypeError('wrong metric')
        if type(weights) != bool:
            raise TypeError('wrong weights type')
        if type(test_block_size) != int:
            raise TypeError('wrong test_block_size type')

        self.k = k
        self.strategy = strategy
        if self.strategy != 'my_own':
            self.sklearn_Classifier = NearestNeighbors(n_neighbors=k, 
                                                       algorithm=strategy,
                                                       metric=metric)
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
    
    def fit(self, X, y):

        self.data = X
        self.answers = y

        if X.shape[0] != y.shape[0]:
            raise TypeError('answers size dont correspond data size')
        if self.strategy != 'my_own':
            self.sklearn_Classifier.fit(X, y)

    def find_kneighbors(self, X, return_distance=False):

        if self.metric == 'euclidean':
            pairwise_matr = distances.euclidean_distance(X, self.data)
        else:
            pairwise_matr = distances.cosine_distance(X, self.data)

        if return_distance:
            return np.sort(pairwise_matr, axis=1)[:, :self.k], np.argsort(pairwise_matr, axis=1)[:, :self.k]
        else:
            return np.argsort(pairwise_matr, axis=1)[:, :self.k]

    def cycle(self, func, X, test_block_size, return_tuple=False):

        return_arr_idxs = np.empty(shape=(X.shape[0], self.k), dtype='int64')

        if return_tuple:
            return_arr = np.empty(shape=(X.shape[0], self.k), dtype='float64')

        for block_idx in range(X.shape[0] // test_block_size):
            if return_tuple:
                return_arr[block_idx * test_block_size:(block_idx + 1) * test_block_size] = func(X[block_idx * test_block_size:(block_idx + 1) * test_block_size], True)[0]
                return_arr_idxs[block_idx * test_block_size:(block_idx + 1) * test_block_size] = func(X[block_idx * test_block_size:(block_idx + 1) * test_block_size], True)[1]
            else:
                return_arr_idxs[block_idx * test_block_size:(block_idx + 1) * test_block_size] = func(X[block_idx * test_block_size:(block_idx + 1) * test_block_size], False)
        
        if X.shape[0] % test_block_size:
            block_idx = X.shape[0] // test_block_size

            if return_tuple:
                return_arr[block_idx * test_block_size:] = func(X[block_idx * test_block_size:], True)[0]
                return_arr_idxs[block_idx * test_block_size:] = func(X[block_idx * test_block_size:], True)[1]
            else:
                return_arr_idxs[block_idx * test_block_size:] = func(X[block_idx * test_block_size:], False)

        if return_tuple:
            return return_arr, return_arr_idxs
        else:
            return return_arr_idxs   
  
    def predict(self, X):
        self.test = X

        def transform_to_classes(idx):
                return self.answers[idx]

        if self.weights:
            eps = 0.000001
            self.counter = 0 
            if self.strategy != 'my_own':
                just_dists, just_idxs = self.sklearn_Classifier.kneighbors(self.test, self.k, True)
            else:
                just_dists, just_idxs = self.cycle(self.find_kneighbors, X, self.test_block_size, True)

            def string_to_ans(s):
                self.counter += 1
                return np.argmax(np.bincount(s, weights=(1 / (just_dists + eps))[self.counter-1]))

            return np.apply_along_axis(string_to_ans, 1, transform_to_classes(just_idxs))
        else:
            if self.strategy != 'my_own':
                just_idxs = self.sklearn_Classifier.kneighbors(self.test, self.k, True)

            just_idxs = self.cycle(self.find_kneighbors, X, self.test_block_size)
            return np.apply_along_axis(lambda s: np.argmax(np.bincount(s)), 1, transform_to_classes(just_idxs))
