import matplotlib.pyplot as plt
from pylab import *

x = [1,2,3]
print(len(x))
y = [5,7,4]

x2 = [1,2,3]
y2 = [10,14,12]

plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')

plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
# 然后我们可以使用plt.legend()生成默认图例。
plt.show()

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.cos(X)*(-1)

plot(X,C)
plot(X,S)
show()

import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
