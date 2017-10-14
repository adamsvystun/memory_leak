import numpy as np
import matplotlib.pyplot as plt

def normalize_day(dist):
    dist2 = []
    dist3 = []
    for i in range(1,100):
        dist2.append(0)
        dist3.append(0)
    for i in range(0,len(dist)):
        dist2[dist[i]] += 1
    for i in range(1,25):
        min_i = i
        while dist2[i] > 0:
            dist2[i] -= 1
            while dist3[min_i] == 1:
                min_i += 1
                if min_i == 25:
                    min_i = 1
            dist3[min_i] = 1
    count = 0
    for i in range(1,25):
        if dist3[i] == 1:
            dist[count] = i
            count += 1
    return dist

dist=[2,2,3,5,5,5,7,8,8,15]
print(normalize_day(dist))
