from utils import dataloader
from utils import normalizer
import sys
import math
import pdb
import numpy as np
class LinearRegression:

    def __init__(self, d = 1, lr = 0.001, batch_size = 32, stopping_criteria = 1e-9, max_steps = 1e4, k = 1, normalize = False):

        self.d = d
        self.lr = lr
        self.batch_size = batch_size
        self.eps = stopping_criteria
        self.theta = np.zeros((d+1,1))
        self.max_steps = max_steps
        self.k = k
        self.normalize = normalize
    
    def fit(self, x, y):
        '''
        Args:
            X: numpy array of shape [N x d]
            Y: numpy array of shape [N x 1]
        Returns:
            dict:{
                theta: [d x 1],
                steps: [steps x d+1]
            }
        '''
        #normalize data 
        if(self.normalize):
            x, self.x_mu, self.x_sigma = normalizer(x)
        
        #add intercept term
        #X: [N x (d+1)]
        m = x.shape[0]
        one = np.ones((m, 1))
        x = np.concatenate((x, one), axis = 1)

        #adjust the batch size
        batch_size = self.batch_size
        if(batch_size == -1):
            batch_size = m

        #inital loss and old loss
        j_theta = ((x.dot(self.theta) - y)**2).mean()/2
        j_theta_old = [j_theta + self.eps + 1 for i in range(self.k)]

        #track the steps
        steps = [(self.theta.copy(), j_theta)]

        #track the batches
        curr_batch = 0
        total_batches = math.ceil(m/batch_size)
        
        #loop over updates
        print(j_theta_old[-1], j_theta)
        while sum(j_theta_old)/self.k - j_theta > self.eps:
            #get the batch data
            sb = curr_batch*batch_size
            eb = sb + batch_size
            x_batch = x[sb:eb]
            y_batch = y[sb:eb]

            #compute gradient
            y_pred = x_batch.dot(self.theta)
            gradient = ((x_batch.T).dot(y_pred - y_batch))/batch_size
            
            #update theta
            self.theta = self.theta - self.lr*gradient
            
            #compute j theta
            j_theta_old.append(j_theta)
            j_theta_old = j_theta_old[1:]
            j_theta = ((x.dot(self.theta) - y)**2).mean()/2
            steps.append((self.theta.copy(), j_theta))

            #update batch
            curr_batch += 1
            curr_batch = curr_batch % total_batches
            print(j_theta_old[-1], j_theta)
            
        return {'theta': self.theta.copy(), 'steps': steps}


if __name__ == '__main__':
    x,y = dataloader(sys.argv[1], sys.argv[2])
    linear_regression = LinearRegression(batch_size = -1)
    results = linear_regression.fit(x,y)
    print(results)

