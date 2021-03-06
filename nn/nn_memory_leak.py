import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json

np.random.seed(10)

epochs = 2000
batch_size = 64
no_hidden1 = 50 #num of neurons in hidden layer 1
learning_rate = 0.0001
print(batch_size)
print(no_hidden1)
print(learning_rate)

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
X_data, Y_data = book_rental_history[:,:6], book_rental_history[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# scale and normalize data
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

trainX = scale(trainX, trainX_min, trainX_max)
testX = scale(testX, testX_min, testX_max)

trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std)
testX = normalize(testX, testX_mean, testX_std)

no_features = trainX.shape[1]
x = T.matrix('x') # data sample
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

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )


train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)


min_error = 1e+15
best_iter = 0
best_w_o = np.zeros(no_hidden1)
best_w_h1 = np.zeros([no_features, no_hidden1])
best_b_o = 0
best_b_h1 = np.zeros(no_hidden1)

alpha.set_value(learning_rate)

n = len(trainX)
print(n)
t = time.time()
for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    trainX, trainY = shuffle_data(trainX, trainY)
    cost = 0.0
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
        cost += train(trainX[start:end], np.transpose(trainY[start:end]))
    train_cost[iter] = cost/(n // batch_size)
    pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

    if test_cost[iter] < min_error:
        best_iter = iter
        min_error = test_cost[iter]
        best_w_o = w_o.get_value()
        best_w_h1 = w_h1.get_value()
        best_b_o = b_o.get_value()
        best_b_h1 = b_h1.get_value()

#set weights and biases to values at which performance was best
w_o.set_value(best_w_o)
b_o.set_value(best_b_o)
w_h1.set_value(best_w_h1)
b_h1.set_value(best_b_h1)
best_nn = [best_w_o.tolist(), best_b_o.tolist(), best_w_h1.tolist(), best_b_h1.tolist(), trainX_max.tolist(), trainX_min.tolist(), trainX_mean.tolist(), trainX_std.tolist()]

best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

#Plots
plt.figure(1,figsize=(15,9))

plt.subplot(121)
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('iterations')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.2g'%learning_rate)

plt.subplot(122)
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('p_1b_Training_Test_Errors_and_Accuracy_n_batch_32_decay_10**5_neurons_60_fin.png')
plt.show()

with open("best_nn.txt", "w") as fp:   #Pickling
    fp.write(json.dumps(best_nn))
