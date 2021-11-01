import numpy as np

def compute_mse(y, tx, w):
    error= y-(tx@w)
    N = len(y)
    mse = error.dot(error)/(2*N)

    return mse

def compute_rmse(y, tx, w):
    rmse = np.sqrt(2*compute_mse(y,tx, w))
    return rmse
    