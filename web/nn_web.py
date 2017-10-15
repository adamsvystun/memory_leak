import time
import numpy as np
import theano
import theano.tensor as T

import pickle
import json

np.random.seed(10)

no_hidden1 = 60 #num of neurons in hidden layer 1

with open("test.json", "r") as fp:
    book_rental_h = json.load(fp)

with open("best_nn.txt", "r") as fp:   # Unpickling
    best_nn = json.load(fp)

floatX = theano.config.floatX
# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

#read and divide data into test and train sets
x = T.vector('x') # data sample
d = T.matrix('d') # desired output

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(6, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

result = theano.function(
    inputs = [x],
    outputs = [y],
    allow_input_downcast=True
)

#set weights and biases to values at which performance was best
w_o.set_value(best_nn[0])
b_o.set_value(best_nn[1])
w_h1.set_value(best_nn[2])
b_h1.set_value(best_nn[3])

def query(pk):
    book = book_rental_h[pk]
    author = book[0]
    type_b = book[2]
    year_time = [0]*52
    week_time = [0]*7
    for h in range(8736):
        current_hour = h % 24
        day = ((h - current_hour) / 24)
        current_day_of_week = int(day % 7)
        current_week = int((day - current_day_of_week) / 7)
        x = [author, pk, current_week, current_day_of_week, current_hour, type_b]
        x = scale(x, np.array(best_nn[5]), np.array(best_nn[4]))
        x = normalize(x, np.array(best_nn[6]), np.array(best_nn[7]))
        r = result(x)[0]
        year_time[current_week] += r
        week_time[current_day_of_week] += r
    for i in range(7):
        week_time[i] /= 52*24
    for i in range(52):
        year_time[i] /=  24*7
    return {
        "week": week_time,
        "year": year_time,
        "week_old": book_rental_h[pk][5],
        "year_old": book_rental_h[pk][6],
    }
    
