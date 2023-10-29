import sys 
import pdb
from linear_regression import LinearRegression
from utils import dataloader, normalizer
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm

if __name__ == '__main__':
    
    #load the data and run the linear regressor
    x,y = dataloader(sys.argv[1], sys.argv[2])
    lr = float(sys.argv[3])
    linear_regression = LinearRegression(batch_size = -1, lr = lr, normalize = True)
    results = linear_regression.fit(x,y)
    
    #part a
    print('theta: ', results['theta'])
    print('lr: ', linear_regression.lr)
    print('eps: ', linear_regression.eps)
    
    #part b
    x, _, _ = normalizer(x)
    plt.scatter(x[:,0], y[:,0])
    x_ = np.linspace(4, -3, 100)
    plt.plot(x_, x_*results['theta'][0]+ results['theta'][1], linestyle='solid', color='green')
    plt.show()

    #part c
    theta_1 = np.arange(-1, 1, 2/1000)
    theta_2 = np.arange(0, 2, 2/1000)
    theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
    y_pred = x[:,0]*(theta_1.reshape(1000,1000,1)) + (theta_2.reshape(1000,1000,1))
    j_theta = ((y[:,0] - y_pred)**2).mean(axis = -1)/2
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(theta_1, theta_2, j_theta, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    ax.plot_wireframe(theta_1, theta_2, j_theta, rstride=30, cstride=30)
    theta_1 = [t[0][0,0] for t in results['steps']]
    theta_2 = [t[0][1,0] for t in results['steps']]
    j_theta = [t[1] for t in results['steps']]
    ax.scatter(theta_1, theta_2, j_theta,  color='green')
    plt.show()

    #part d
    theta_1 = np.arange(-1, 1, 2/1000)
    theta_2 = np.arange(0, 2, 2/1000)
    theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
    y_pred = x[:,0]*(theta_1.reshape(1000,1000,1)) + (theta_2.reshape(1000,1000,1))
    j_theta = ((y[:,0] - y_pred)**2).mean(axis = -1)/2
    
    plt.contour(theta_1, theta_2, j_theta)
    theta_1 = [t[0][0,0] for t in results['steps']]
    theta_2 = [t[0][1,0] for t in results['steps']]
    plt.scatter(theta_1, theta_2,  color='green', s=1)
    plt.show()
    
#CHASAN@2231
