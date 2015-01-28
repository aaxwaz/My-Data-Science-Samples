# __author__: weimin
# this script demonstrates a simple neuron network to implement function
# f = sigmoid(a*x + b*y + c). Both forward and backward propagations are implemented
# Original source from Andrej Karpathy blog: http://karpathy.github.io/neuralnets/

import math

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
        
class sigmoidGate(object):
    def sig(self, x):
        return 1/(1 + math.exp(-x))
        
    def forwardProp(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop
        
    def backwardProp(self):
        temp = self.sig(self.u0.value)
        self.u0.grad += temp * (1-temp) * self.utop.grad
        
def fastForwardProp(a, b, c, x, y):
    return 1/(1 + math.exp(-(a*x + b*y +c)))
    
def forwardProp():
    ax = multi0.forwardProp(a, x)
    by = multi1.forwardProp(b, y)
    axpby = add0.forwardProp(ax, by)
    axpbypc = add1.forwardProp(axpby,c)
    s = sigmoid0.forwardProp(axpbypc)
    return s
    
def backwardProp():
    sigmoid0.backwardProp()
    add1.backwardProp()
    add0.backwardProp()
    multi1.backwardProp()
    multi0.backwardProp()
    
multi0 = multiGate()
multi1 = multiGate()
add0 = addGate()
add1 = addGate()
sigmoid0 = sigmoidGate()
        
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)
    
# testing 
# analytics gradients using back-prop
s = forwardProp()
print "s value and gradient are: "
print s.value, s.grad
s.grad = 1.0
backwardProp()
print 
print "Now increase the gradient of s by 1"
print 
print "new analytical gradients after 1 round of back-prop "
print a.grad, b.grad, c.grad, x.grad, y.grad

#numeric result to check the correctness
h = 0.0001
a = a.value
b = b.value
c = c.value
x = x.value
y = y.value

a_grad = (fastForwardProp(a+h,b,c,x,y) - fastForwardProp(a,b,c,x,y))/h;
b_grad = (fastForwardProp(a,b+h,c,x,y) - fastForwardProp(a,b,c,x,y))/h;
c_grad = (fastForwardProp(a,b,c+h,x,y) - fastForwardProp(a,b,c,x,y))/h;
x_grad = (fastForwardProp(a,b,c,x+h,y) - fastForwardProp(a,b,c,x,y))/h;
y_grad = (fastForwardProp(a,b,c,x,y+h) - fastForwardProp(a,b,c,x,y))/h;
        
print 
print "numeric gradients are: "
print a_grad, b_grad, c_grad, x_grad, y_grad
        
        
        
        
        
