from matplotlib import pyplot as mp
import numpy as np

from .math import gaussian, gaussian_f

def main():
    number_of_points = 70
    # 8736
    g = gaussian_f(np.linspace(0, 100, number_of_points), 50, 10)
    # mp.plot()
    d = []
    d.append(0)
    for i, point in enumerate(g):
        d.append(d[i] + point)
    d = d[:number_of_points]
    mp.plot(d, np.zeros_like(d), "x")
    mp.show()

if __name__ == "__main__":
    main()
