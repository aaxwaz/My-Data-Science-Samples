# This simple 2-layer neural network implements below function:
# n1 = max(0, a1*x + b1*y + c1); 
# n2 = max(0, a2*x + b2*y + c2); 
# n3 = max(0, a3*x + b3*y + c3)
# score = a4*n1 + b4*n2 + c4*n3 + d4

import math
import random

#training data set with labels +1 or -1
data = []; labels = []
data.append([1.2, 0.7]); labels.append(1);
data.append([-0.3, -0.5]); labels.append(-1);
data.append([3.0, 0.1]); labels.append(1);
data.append([-0.1, -1.0]); labels.append(-1);
data.append([-1.0, 1.1]); labels.append(-1);
data.append([2.1, -3]); labels.append(1);

#initialization 
a1 = random.random() - 0.5
b1 = random.random() - 0.5
c1 = random.random() - 0.5
a2 = random.random() - 0.5
b2 = random.random() - 0.5
c2 = random.random() - 0.5
a3 = random.random() - 0.5
b3 = random.random() - 0.5
c3 = random.random() - 0.5
a4 = random.random() - 0.5
b4 = random.random() - 0.5 
c4 = random.random() - 0.5
d4 = random.random() - 0.5

def evalTrainingAccuracy():  
    num_correct = 0
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]       
        true_label = labels[i]
        
        n1 = max(0, a1*x + b1*y + c1)
        n2 = max(0, a2*x + b2*y + c2)
        n3 = max(0, a3*x + b3*y + c3)
        score = a4*n1 + b4*n2 + c4*n3 + d4
        
        predicted_label = 1.0 if score >0 else -1.0
        
        if(predicted_label == true_label):
            num_correct += 1
    return float(num_correct) / len(data)

for iter in range(4000):

    # pick a random data point
    i = int(math.floor(random.random() * len(data)))
    x = data[i][0]
    y = data[i][1]
    label = labels[i]
    
    # compute forward pass
    n1 = max(0, a1*x + b1*y + c1)
    n2 = max(0, a2*x + b2*y + c2)
    n3 = max(0, a3*x + b3*y + c3)
    score = a4*n1 + b4*n2 + c4*n3 + d4
    
    # compute pull on the top
    pull = 0.0
    if label == 1 and score < 1:
        pull = 1
    if label == -1 and score > -1:
        pull = -1
        
        
    # now compute backward pass to all parameters of the model 
    dscore = pull
    da4 = n1 * dscore
    dn1 = a4 * dscore
    db4 = n2 * dscore
    dn2 = b4 * dscore
    dc4 = n3 * dscore
    dn3 = c4 * dscore
    dd4 = 1.0 * dscore
    
    # just set the gradients to zero if the neurons did not "fire"
    dn3 = 0 if n3 == 0 else dn3
    dn2 = 0 if n2 == 0 else dn2
    dn1 = 0 if n1 == 0 else dn1
    
    # backprop to parameters of neuron 1
    da1 = x * dn1
    db1 = y * dn1
    dc1 = 1.0 * dn1

    # backprop to parameters of neuron 2
    da2 = x * dn2
    db2 = y * dn2
    dc2 = 1.0 * dn2
    
    # backprop to parameters neuron 3
    da3 = x * dn3
    db3 = y * dn3
    dc3 = 1.0 * dn3

    # regularization
    da1 += -a1; da2 += -a2; da3 += -a3
    db1 += -b1; db2 += -b2; db3 += -b3
    da4 += -a4; db4 += -b4; dc4 += -c4

    # parameters update
    step_size = 0.01
    a1 += step_size * da1 
    b1 += step_size * db1 
    c1 += step_size * dc1
    a2 += step_size * da2 
    b2 += step_size * db2
    c2 += step_size * dc2
    a3 += step_size * da3 
    b3 += step_size * db3 
    c3 += step_size * dc3
    a4 += step_size * da4 
    b4 += step_size * db4 
    c4 += step_size * dc4 
    d4 += step_size * dd4

    if iter % 25 == 0:
        print 'training accuracy at iter %d : %f' % (iter, evalTrainingAccuracy())






























































