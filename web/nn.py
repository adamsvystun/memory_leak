import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json

np.random.seed(10)

epochs = 1000
no_hidden1 = 50 #num of neurons in hidden layer 1

with open("test.json", "r") as fp:
    book_rental_h = json.load(fp)

with open("best_nn.txt", "rb") as fp:   # Unpickling
    best_nn = pickle.load(fp)

floatX = theano.config.floatX
#print(book_rental_h[1][3])
# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

#read and divide data into test and train sets
x = T.vector('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(6, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )
result = theano.function(
    inputs = [x],
    outputs = [y],
    allow_input_downcast=True
    )


rental_time = []
min_error = 1e+15
best_iter = 0

#set weights and biases to values at which performance was best
w_o.set_value(best_nn[0])
b_o.set_value(best_nn[1])
w_h1.set_value(best_nn[2])
b_h1.set_value(best_nn[3])

author = 0
ide = 300
type_b = 1
rental_time = np.zeros(52)
for h in range(8736):
    current_hour = h % 24
    day = ((h - current_hour) / 24)
    current_day_of_week = int(day % 7)
    current_week = int((day - current_day_of_week) / 7)
    x = [author, ide, current_week, current_day_of_week, current_hour, type_b]
    x = scale(x, best_nn[5], best_nn[4])
    x = normalize(x, best_nn[6], best_nn[7])
    rental_time[current_week] += result(x)


#Plots
plt.figure(1,figsize=(15,9))

plt.subplot(121)
plt.plot(book_rental_h[0][6])
plt.xlabel('time line')
plt.ylabel('Demands from the data set')
plt.title('History of the demands at time')
plt.subplot(122)
plt.plot(range(52), rental_time, label='rental time')
plt.xlabel('time line')
plt.ylabel('Rental time')
plt.title('Rental time given by the network')
plt.savefig('test_bb.png')
plt.show()
