import numpy as np


def euclidean_distance(X, Y):
    # X ~ (N, D)
    # Y ~ (M, D)
    X_squares = (X ** 2).sum(axis=1) # ~ N
    Y_squares = (Y ** 2).sum(axis=1) # ~ M
    return np.sqrt(X.dot(Y.T) * (-2) +
                   X_squares[None, :].T + Y_squares)
def cosine_distance(X, Y):
    # X ~ (N, D)
    # Y ~ (M, D)
    return 1 - \
           X.dot(Y.T) / (np.linalg.norm(X, axis=1)[None, :].T * np.linalg.norm(Y, axis=1))
