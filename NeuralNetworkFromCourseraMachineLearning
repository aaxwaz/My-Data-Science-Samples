import numpy as np
import random
from scipy import optimize

class NeuralNetwork(object):

    def __init__(self, epsilon_init = 0.12, reg_lambda = 1.0, opti_method='TNC', hidden_layer_size = 25, maxiter=200):
        self.epsilon_init = epsilon_init
        self.opti_method = opti_method
        self.maxiter = maxiter
        self.hidden_layer_size = hidden_layer_size
        self.reg_lambda = reg_lambda
    
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x));
        
    def randInitializeWeights(self, L_in, L_out):
        W = np.ones([L_out, L_in + 1])
        W = W * random.random() * 2 * self.epsilon_init - self.epsilon_init
        return W
    
    def feedForward(self):
        pass
    
    def nnCostFunction(self, nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda):
        # nnCostFunction is used to calculate cost and gradients
        # parameters: 
        # theta1, theta2: one dimensional arrays containing weights for the network
        # input_layer_size: integer, indicating size of input layer
        # hidden_layer_size: integer, indicating size of hidden layer
        # num_labels: integer, indicating size of output layer
        # X, y: training and target data set; X is m by n (e.g. 5000 x 400) where y is m by 1 (e.g. from 1 to 10)
        # theLambda: regularization coefficient
        theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)] 
        theta1 = theta1.reshape(hidden_layer_size, input_layer_size + 1)
        theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
        theta2 = theta2.reshape(num_labels, hidden_layer_size + 1)
        theta1_grad = np.zeros(np.shape(theta1)) # 25 x 401
        theta2_grad = np.zeros(np.shape(theta2)) # 10 x 26
         
        m = np.size(X,0)                              # 5000
        n = np.size(X,1)                              # 400
        J = 0.0
        
        # initializing
        y = y.reshape(m, )
        y = y-1
        y = np.eye(num_labels)[y]                     # 5000 x 10
        a1 = np.append(np.ones([m,1]),X,1)            # 5000 x 401
        
        # feedForward
        z2 = np.dot(theta1, a1.T)                     # 25 x 5000
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones([1,m]), a2, 0)         # 26 x 5000
        
        z3 = np.dot(theta2, a2)
        a3 = self.sigmoid(z3).T                       # 5000 * 10
        
        J = -y * np.log(a3) - (1-y) * np.log(1- a3)
        J = np.sum(J)
        
        # back-prop
        delta_3 = a3 - y            # 5000 x 10
        delta_2 = np.dot(theta2.T, delta_3.T) * (a2 * (1-a2))    # 26 x 5000    
        delta_2 = delta_2[1:,:]   # 25 x 5000        
        theta2_grad = theta2_grad + np.dot(delta_3.T, a2.T)
        theta1_grad = theta1_grad + np.dot(delta_2, a1)
        
        '''
        for i in range(m):
            target_of_i = y_matrix[i,:].reshape(1,num_labels)                                  # 1 x 10
            
            a1 = X_matrix[i,:].reshape(1,n+1)                                                  # 1 x 401
            z2 = np.dot(a1,np.transpose(theta1)).reshape(1,hidden_layer_size)                  # 1 x 25
            a2 = self.sigmoid(z2)                                                              # 1 x 25
            a2 = np.append(np.array(1.0).reshape(1,1),a2,1).reshape(1,hidden_layer_size + 1)   # 1 x 26
            z3 = np.dot(a2,np.transpose(theta2))                                               # 1 x 10
            a3 = self.sigmoid(z3)                                                              # 1 x 10
            
            # backprop 
            delta_3 = (a3 - target_of_i).reshape(1,num_labels)   # 1 x 10
            delta_2 = np.dot(np.transpose(theta2), np.transpose(delta_3)) * (a2 * (1 - a2)).reshape(hidden_layer_size + 1, 1) # 26 x 1
            delta_2 = delta_2[1:]
            theta2_grad = theta2_grad + np.dot(np.transpose(delta_3), a2)
            theta1_grad = theta1_grad + np.dot(delta_2, a1)
            
            for k in range(num_labels):
                J = J - ( target_of_i[0,k] * np.log(a3[0,k])  + (1.0 - target_of_i[0,k]) * np.log(1.0 - a3[0,k]) )
        '''
        
        J = 1.0 / m * J
        theta1_grad = theta1_grad / float(m)
        theta2_grad = theta2_grad / float(m)
        
        # regularization for gradients
        theta1_grad[:,1:] = theta1_grad[:,1:] + float(theLambda)/m*(theta1[:,1:])
        theta2_grad[:,1:] = theta2_grad[:,1:] + float(theLambda)/m*(theta2[:,1:])
        
        # regularization for cost J
        tempTheta1 = np.power(theta1,2)
        tempTheta2 = np.power(theta2,2)
        tempSum = sum(sum(tempTheta1[:,1:]))
        tempSum = tempSum + sum(sum(tempTheta2[:,1:]))
        J = J + float(theLambda)/(2*m)*tempSum
        
        grad = np.append(theta1_grad.ravel(), theta2_grad.ravel())
        
        return(J, grad)
        
    def fit(self, X, y):
        #initializing:
        input_layer_size = np.size(X,1)
        num_labels = len(set(y.reshape(np.size(X,0),)))
        theta1 = self.randInitializeWeights(input_layer_size, self.hidden_layer_size)   # 25 x 401
        theta2 = self.randInitializeWeights(self.hidden_layer_size, num_labels)         # 10 x 26
        nn_params = np.append(theta1.ravel(), theta2.ravel())
        options = {'maxiter' : self.maxiter}
        
        #running optimize function
        res = optimize.minimize(self.nnCostFunction, nn_params, jac = True, method = self.opti_method, options = options, 
        args = (input_layer_size, self.hidden_layer_size, num_labels, X, y, self.reg_lambda))
        print "optimization finished! Message: "
        print res.message
        self.t1 = (res.x[0:(self.hidden_layer_size * (input_layer_size+1))]).reshape(self.hidden_layer_size, input_layer_size+1)
        self.t2 = (res.x[(self.hidden_layer_size * (input_layer_size+1)):]).reshape(num_labels, self.hidden_layer_size+1)    
        
    def predict(self, X):
        m = np.size(X,0)                              # 5000
        
        # initializing
        a1 = np.append(np.ones([m,1]),X,1)            # 5000 x 401
        
        # feedForward
        z2 = np.dot(self.t1, a1.T)                    # 25 x 5000
        a2 = self.sigmoid(z2)
        a2 = np.append(np.ones([1,m]), a2, 0)         # 26 x 5000
        
        z3 = np.dot(self.t2, a2)
        a3 = self.sigmoid(z3).T                       # 5000 * 10
            
        result = (a3.argmax(1) + 1).reshape(m, 1)
        return result
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
