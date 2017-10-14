from matplotlib import pyplot as mp
import numpy as np

from .math import gaussian

def main():
    g = [1, 2, 3, 7, 9, 14, 24, 37]
    mp.plot(g, np.zeros_like(g), "x")
    mp.show()

if __name__ == "__main__":
    main()
