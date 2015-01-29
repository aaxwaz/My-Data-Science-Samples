# This script implemented a svm classifier trained on a set of N = 6 data points
# Original source from Andrej Karpathy blog: http://karpathy.github.io/neuralnets/

import math
import random

class Unit(object):
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad
        
class multiGate(object):
    def forwardProp(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit( (self.u0.value * self.u1.value), 0.0 )
        return self.utop
        
    def backwardProp(self):
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad
        
class addGate(object):
    def forwardProp(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit( (self.u0.value + self.u1.value), 0.0 )
        return self.utop
        
    def backwardProp(self):
        self.u0.grad += self.utop.grad * 1
        self.u1.grad += self.utop.grad * 1
        
        
class Circuit(object):
    def __init__(self):
        self.mulg0 = multiGate() 
        self.mulg1 = multiGate() 
        self.addg0 = addGate() 
        self.addg1 = addGate() 
        
    def forwardProp(self, x, y, a, b, c):
        self.ax = self.mulg0.forwardProp(a, x)  
        self.by = self.mulg1.forwardProp(b, y)  
        self.axpby = self.addg0.forwardProp(self.ax, self.by)  
        self.axpbypc = self.addg1.forwardProp(self.axpby, c)  
        return self.axpbypc 
        
    def backwardProp(self, gradient_top):
        self.axpbypc.grad = gradient_top 
        self.addg1.backwardProp()  
        self.addg0.backwardProp()  
        self.mulg1.backwardProp()  
        self.mulg0.backwardProp() 
        
        
class SVM(object):
    def __init__(self):
        self.a = Unit(1.0, 0.0)
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(-1.0, 0.0)
        self.circuit = Circuit()
        
    def forwardProp(self, x, y):
        self.unit_out = self.circuit.forwardProp(x, y, self.a, self.b, self.c)
        return self.unit_out
        
    def backwardProp(self, label):
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0
        
        self.pull = 0.0
        
        if (label == 1) and (self.unit_out.value < 1):
            self.pull = 1.0
            
        if (label == -1) and (self.unit_out.value > -1):
            self.pull = -1.0
            
        self.circuit.backwardProp(self.pull)
        
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value
        
    def learnFrom(self, x, y, label):
        self.forwardProp(x, y)
        self.backwardProp(label)
        self.parameterUpdate()
        
    def parameterUpdate(self):
        step_size = 0.01
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad
        
data = []; labels = []
data.append([1.2, 0.7]); labels.append(1);
data.append([-0.3, -0.5]); labels.append(-1);
data.append([3.0, 0.1]); labels.append(1);
data.append([-0.1, -1.0]); labels.append(-1);
data.append([-1.0, 1.1]); labels.append(-1);
data.append([2.1, -3]); labels.append(1);
   
svm = SVM()   

def evalTrainingAccuracy():  
    num_correct = 0
    for i in range(len(data)):
        x = Unit(data[i][0], 0.0)
        y = Unit(data[i][1], 0.0)        
        true_label = labels[i]
        
        predicted_label = 1.0 if svm.forwardProp(x,y).value >0 else -1.0
        
        if(predicted_label == true_label):
            num_correct += 1
    return float(num_correct) / len(data)
        
              
for iter in range(400):

    i = int(math.floor(random.random() * len(data)))
    x = Unit(data[i][0], 0.0)
    y = Unit(data[i][1], 0.0)
    label = labels[i]
    svm.learnFrom(x,y,label)
    
    if iter % 25 == 0:
        print 'training accuracy at iter %d : %f' % (iter, evalTrainingAccuracy())
        
        
  
        
        
        
        
        
        
        
        
        
        
        



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
