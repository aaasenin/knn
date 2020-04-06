import numpy as np
from nearest_neighbors import *


def kfold(n, n_folds):    
    return_list = []

    if not n % n_folds:
        block_size = n // n_folds
        block_idx = 0

        return_list.append((np.arange((block_idx + 1) * block_size, n),
                           np.arange(block_idx * block_size, (block_idx + 1) * block_size)))

        for block_idx in range(1, n_folds - 1):
            return_list.append((np.append(np.arange(block_idx * block_size), np.arange((block_idx + 1) * block_size, n)),
                                np.arange(block_idx * block_size, (block_idx + 1) * block_size)))

        block_idx = n_folds - 1
        return_list.append((np.arange(block_idx * block_size),
                           np.arange(block_idx * block_size, (block_idx + 1) * block_size)))
    else:
        block_size = n // n_folds
        block_idx = 0

        return_list.append((np.arange((block_idx + 1) * block_size, n),
                           np.arange(block_idx * block_size, (block_idx + 1) * block_size)))

        for block_idx in range(1, n_folds - 1):
            return_list.append((np.append(np.arange(block_idx * block_size), np.arange((block_idx + 1) * block_size, n)),
                               np.arange(block_idx * block_size, (block_idx + 1) * block_size)))
        block_idx = n_folds - 1
        return_list.append((np.arange(block_idx * block_size),
                           np.arange(block_idx * block_size, n)))

    return return_list

def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):   
    ret_dict = dict.fromkeys(k_list,)

    for key in ret_dict.keys():
        ret_dict[key] = np.array([])

    if score != 'accuracy':
        raise TypeError("unknown score")
    if cv == None:
        cv = kfold(X.shape[0], 3)

    my_Classifier = KNNClassifier(k=k_list[-1], **kwargs) 
    for train_idxs, test_idxs in cv:
        my_Classifier.fit(X[train_idxs], y[train_idxs])

        if my_Classifier.weights:
            eps = 0.000001
            just_dists, just_idxs = my_Classifier.find_kneighbors(X[test_idxs], True)
            vv = y[train_idxs][just_idxs]

            for k in k_list:
                k_just_dists = just_dists[:,:k]
                true_ans = []

                for jdx, v in enumerate(vv[:,:k]):
                    tmp = np.bincount(v, weights=(1 / (k_just_dists[jdx] + eps)))
                    true_ans.append(np.argmax(tmp))
                ret_dict[k] = np.append(ret_dict[k], np.sum(true_ans == y[test_idxs]) / len(y[test_idxs]))
        else:
            just_idxs = my_Classifier.find_kneighbors(X[test_idxs], False)
            vv = y[train_idxs][just_idxs]

            for k in k_list:
                true_ans = []
                for jdx, v in enumerate(vv[:,:k]):
                    tmp = np.bincount(v)
                    true_ans.append(np.argmax(tmp))
                ret_dict[k] = np.append(ret_dict[k], np.sum(true_ans == y[test_idxs]) / len(y[test_idxs]))

    return ret_dict
