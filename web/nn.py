import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

np.random.seed(10)

epochs = 1000
batch_size = 64
no_hidden1 = 40 #num of neurons in hidden layer 1
learning_rate = 0.0001

with open("best_nn.txt", "rb") as fp:   # Unpickling
    best_nn = pickle.load(fp)

floatX = theano.config.floatX

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

book_rental_history = np.loadtxt('data/input.data', delimiter=',')
X_data, Y_data = book_rental_history[:901,:6], book_rental_history[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data

testX, testY = X_data,Y_data

# scale and normalize data
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

testX = scale(testX, testX_min, testX_max)

testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

testX = normalize(testX, testX_mean, testX_std)

no_features = testX.shape[1]
x = T.vector('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

# learning rate
alpha = theano.shared(learning_rate, floatX)


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

rental_time = np.zeros(len(testX))
test_cost = np.zeros(len(testX))
test_accuracy = np.zeros(len(testX))

min_error = 1e+15
best_iter = 0

alpha.set_value(learning_rate)

#set weights and biases to values at which performance was best
w_o.set_value(best_nn[0])
b_o.set_value(best_nn[1])
w_h1.set_value(best_nn[2])
b_h1.set_value(best_nn[3])

for iter in range(len(testX)):
    rental_time[iter], test_cost[iter], test_accuracy[iter]= test(testX[iter], np.transpose(testY[iter]))

#Plots
plt.figure(1,figsize=(15,9))

plt.subplot(121)
plt.plot(range(len(testX)), test_cost, label='test cost')
plt.plot(range(len(testX)), test_accuracy, label = 'test accuracy')
plt.xlabel('time line')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.2g'%learning_rate)

plt.subplot(122)
plt.plot(range(len(testX)), rental_time, label='rental time')
plt.xlabel('time line')
plt.ylabel('Rental time')
plt.title('Rental Time')
plt.savefig('test_bb.png')
plt.show()
