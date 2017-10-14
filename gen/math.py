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
    d = []
    d.append(0)
    for i, point in enumerate(g):
        d.append(d[i] + point)
    d = d[1:]
    rel_scale = scale/d[n-1]
    l = []
    for point in d:
        l.append(point * rel_scale)
    return l

def gaussian_w(mu, sig, n, demands):
    g = gaussian_f(np.linspace(0, 100, n), mu, sig)
    s = np.sum(g)
    scale = demands/s
    return g*scale

def generate_hours(total_demands, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    gen_hours_per_day = []
    gy = gaussian_w(mu1, sigma1, 52, total_demands)
    gy = [ int(x) for x in gy ]
    time_line = []
    for w in range(52):
        gw = gaussian_w(mu2, sigma2, 7, gy[w])
        gw = [ int(x) for x in gw ]
        for d in range(7):
            if gw[d] == 0:
                continue
            gd = gaussian_d(mu3, sigma3, gw[d], 24)
            gd = [ int(x) for x in gd ]
            for h in range(len(gd)):
                time_line.append((w*7 + d)*24 + gd[h])
    return time_line


def random_parameters(mu, sigma, randomness=1):
    rand_mu = np.random.uniform(mu-10*randomness/mu, mu+10*randomness/mu)
    rand_sigma = np.random.uniform(sigma-20*randomness/sigma, sigma+20*randomness/sigma)
    return rand_mu, rand_sigma
