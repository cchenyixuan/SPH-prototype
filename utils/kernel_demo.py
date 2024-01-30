import numpy as np
import matplotlib.pyplot as plt


def wendland_2d(rij, h):
    if rij > h:
        return 0.0
    coefficient = 9 / np.pi / h ** 2
    return coefficient * (1 - rij / h) ** 6 * (35 / 3 * rij / h * rij / h + 6 * rij / h + 1)


x = [i*0.01-1 for i in range(201)]

fig, ax = plt.subplots(1, 2)
ax[0].plot(x, [wendland_2d(abs(a), 1.0) for a in x], )
ax[0].plot(x, [wendland_2d(abs(a), 0.8) for a in x], color="#22c1c3")
ax[0].plot(x, [wendland_2d(abs(a), 0.6) for a in x], color="#fdbb2d")
ax[0].plot(x, [wendland_2d(abs(a), 0.4) for a in x], )
ax[0].legend(["h=1.0", "h=0.8", "h=0.6", "h=0.4", ])
ax[1].plot(x, [wendland_2d(abs(a), 0.0001) for a in x])
ax[1].legend(["h=0.0001"])
fig.suptitle('Wendland Kernel')
plt.show()

plt.plot(x, [wendland_2d(abs(a), 0.0001) for a in x])
plt.legend(["h=0.0001"])
plt.title("Wendland")
plt.show()