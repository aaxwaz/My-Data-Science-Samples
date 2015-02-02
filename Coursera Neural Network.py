# This is the python version for the Coursera course Machine Learning exercise 4 - neural network. 
# The code was originally implemented in Octave, so I "translated" the codes into python
import numpy as np
from scipy import optimize

class MyNeuron(object):
    
    def __init__(self, reg_lambda=1.0, epsilon_init=0.12, hidden_layer_size=25, opti_method='TNC', maxiter=200):
        self.reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method
        self.maxiter = maxiter
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def sumsqr(self, a):
        return np.sum(a ** 2)
    
    def rand_init(self, l_in, l_out):
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init
    
    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    
    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2
   
    def nnCostFunction(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        
        m = X.shape[0]
        Y = np.eye(num_labels)[y]
        t1f = t1[:, 1:]  
        t2f = t2[:, 1:]  
        
        a1 = np.append(np.ones([m,1]),X,1)            
        
        # feedForward
        z2 = np.dot(t1, a1.T)                   
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones([1,m]), a2, 0)        
        
        z3 = np.dot(t2, a2)
        a3 = self.sigmoid(z3)                     
        
        # cost 
        costPositive = -Y * np.log(a3).T
        costNegative = (1 - Y) * np.log(1 - a3).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m
        
        #Back-Prop 
        d3 = a3 - Y.T  
        d2 = np.dot(t2f.T, d3) * self.activation_func_prime(z2)
        
        Delta2 = np.dot(d3, a2.T) 
        Delta1 = np.dot(d2, a1) 
            
        Theta1_grad = (1.0 / m) * Delta1
        Theta2_grad = (1.0 / m) * Delta2
        
        if reg_lambda != 0:
            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (reg_lambda / m) * t1f
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (reg_lambda / m) * t2f
            reg = (self.reg_lambda / (2.0 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))
            J = J + reg
            
        grad = self.pack_thetas(Theta1_grad, Theta2_grad)
        return (J, grad)
        
    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))
        
        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        
        options = {'maxiter': self.maxiter}
        _res = optimize.minimize(self.nnCostFunction, thetas0, jac=True, method=self.method, 
                                 args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, self.reg_lambda), options=options)
        
        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)
    
    def predict(self, X):
        m = np.size(X,0)                              
        
        # initializing
        a1 = np.append(np.ones([m,1]),X,1)           
        
        # feedForward
        z2 = np.dot(self.t1, a1.T)                   
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones([1,m]), a2, 0)         
        
        z3 = np.dot(self.t2, a2)
        a3 = self.sigmoid(z3).T                       
            
        result = (a3.argmax(1)).reshape(m, 1)
        return result
