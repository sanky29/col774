import sys 
import pdb
from linear_regression import LinearRegression
from utils import dataloader, normalizer
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm

#sample data points 
def sample_points():

    x1 = np.random.normal(3,2,1000000)
    x2 = np.random.normal(-1,2,1000000)
    
    #x: [n 2]
    x = np.stack((x1,x2), axis = 1).astype('float')
    theta = np.array([[1],[2]]).astype('float')
    b = 3
    eps = np.random.normal(0,2**0.5)
    y = x.dot(theta) + b + eps

    return x, y




if __name__ == '__main__':
    
    #load the data and run the linear regressor
    x,y = sample_points()
    lr = 0.001
    batch_size = int(sys.argv[1])
    linear_regression = LinearRegression(d = 2, batch_size = batch_size, lr = lr, k = 1000, stopping_criteria=1e-3)
    results = linear_regression.fit(x,y)
    
    #part a
    print('theta: ', results['theta'])
    print('lr: ', linear_regression.lr)
    print('eps: ', linear_regression.eps)
    print('k: ', linear_regression.k)
    
    
    #part d
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    theta_1 = [t[0][0,0] for t in results['steps']]
    theta_2 = [t[0][1,0] for t in results['steps']]
    theta_3 = [t[0][2,0] for t in results['steps']]
    ax.scatter(theta_1, theta_2, theta_3,  color='green', s = 1)
    plt.show()

#CHASAN@2231
