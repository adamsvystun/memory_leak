import numpy as np
from matplotlib import pyplot as mp

def gaussian_f(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_i(x, mu, sig):
    return 1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian(mu, sigma, number_of_points):
    g = np.random.normal(mu, sigma, number_of_points)
    g = [ int(x) for x in g ]
    return g

def gaussian_d(mu, sig, n, scale):
    g = gaussian_i(np.linspace(0, 100, n), mu, sig)
    # mp.plot(g)
    # mp.show()
    d = []
    d.append(0)
    for i, point in enumerate(g):
        d.append(d[i] + point)
    d = d[1:]
    rel_scale = scale/d[n-1]
    for point in d:
        point = point * rel_scale
    return d

def gaussian_w(mu, sig, n, demands):
    g = gaussian_f(np.linspace(0, 100, n), mu, sig)
    s = np.sum(g)
    scale = demands/s
    return g*scale
