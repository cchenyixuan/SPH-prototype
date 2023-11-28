import numpy as np


class PoissonSolver:
    def __init__(self, m, dx):
        self.m = m
        self.T = np.zeros((m, m), dtype=np.float32)
        self.T[0, 0], self.T[0, 1], self.T[-1, -1],self.T[-1, -2] = 2, -1, 2, -1
        for i in range(1, m-1):
            self.T[i, i] = 2.0
            self.T[i, i-1] = -1
            self.T[i, i + 1] = -1
        # T@u + u@T = h^2*f
        self.h = 1/(m+1)
        self.dx = dx
        self.S = np.array([[np.sin((i+1)*(j+1)*np.pi/(m+1))for j in range(m)] for i in range(m)], dtype=np.float32)
        self.D = np.array([4*np.sin(np.pi*(i+1)/2/(m+1))**2 for i in range(m)], dtype=np.float32)

    def solve(self, f):
        G = 4*self.h**2*self.dx**2 * self.S@f@self.S
        X = np.array([[G[i, j]/(self.D[i]+self.D[j]) for j in range(self.m)] for i in range(self.m)], dtype=np.float32)
        V = self.S@X@self.S
        return V
