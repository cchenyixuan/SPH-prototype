import numpy as np
import matplotlib.pyplot as plt


def bessel_2d(x):
    x = abs(x)
    coefficient = 1/(2*np.pi*np.sqrt(2))
    return coefficient * np.exp(-x)*integral(x, ds=0.0001)


def integral(x, ds=0.0001):
    x = abs(x)
    ans = 0.0
    s_list = (0.0001+_*ds for _ in range(int(200/ds)))
    for s in s_list:
        ans += np.exp(-x*s)/np.sqrt(s+s*s/2)*ds
    return ans


x = [0.001*i for i in range(10000)]
print(integral(2.1, 0.0001))
# y2 = [integral(_, 0.01) for _ in x]
plt.plot(x, [bessel_2d(_) for _ in x])
plt.show()
