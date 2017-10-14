import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

index_range = 100

#the outputs is a normal distribution
def gaussian(mu, sigma, x_max, number_of_points):
    g = np.random.normal(mu, sigma, number_of_points)
    g = [ int(x) for x in g ]
    for elem in range(len(g)):
        if g[elem] < 0:
            g[elem]=0
        if g[elem] > x_max:
            g[elem] = x_max
    return g

for index in range(index_range):
