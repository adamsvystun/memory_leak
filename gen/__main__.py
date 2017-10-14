from matplotlib import pyplot as mp
import numpy as np

from .math import gaussian_d

def main():
    number_of_points = 8736
    # 8736
    d = gaussian_d(11, 5, 5, 24)
    print(d)
    mp.plot(d, np.zeros_like(d), "x")
    mp.show()

if __name__ == "__main__":
    main()
