from matplotlib import pyplot as mp
import numpy as np

from .math import gaussian_d, gaussian_w

def main():
    number_of_points = 8736
    # 8736
    d = gaussian_w(50, 10, 52, 200)
    # d = gaussian_d(11, 5, 5, 24)
    print(np.sum(d))
    mp.plot(d)
    mp.show()

if __name__ == "__main__":
    main()
