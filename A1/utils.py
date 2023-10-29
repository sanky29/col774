import numpy as np 
from numpy import genfromtxt


def dataloader(file1, file2):
    x = genfromtxt(file1, delimiter=',')
    if(len(x.shape) == 1):
        x = x.reshape(-1,1)
    y = genfromtxt(file2, delimiter=',')
    if(len(y.shape) == 1):
        y = y.reshape(-1,1)
    return x,y

def normalizer(x):
    '''
    Args:
        x: [N x d] numpy array
    Returns:
        x: [N x d] normalized
        x_mean: [d] numpy array mean of x
        x_sigma: [d] numpy array sigma of x
    '''
    x_mean = x.mean(axis = 0)
    x_sigma = ((x-x_mean)**2).mean(axis = 0)**0.5
    x = (x-x_mean)/x_sigma
    return x, x_mean, x_sigma