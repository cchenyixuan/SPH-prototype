import numpy
import numpy as np


class ShibataSugiyama:
    def __init__(self):
        self.gamma = 1
        self.dimension = 2

        self.n0_norm = 18 * np.pi

        self.a = 1.0
        self.Q = 0.8 * (self.n0_norm + 16 * np.pi) * (np.sqrt(2 * np.pi * (self.n0_norm + 16 * np.pi)) - 8 * np.pi) / (
                    64 * np.pi * self.n0_norm)
        self.lem = 0.1
        self.phi_h = ...
        self.second_moment_const = min(
            self.n0_norm - 16 * np.pi,
            (self.n0_norm + 16 * np.pi) ** 1.5 * (np.sqrt(2 * np.pi * (self.n0_norm + 16 * np.pi)) - 8 * np.pi) * (np.log((self.n0_norm + 16 * np.pi) / (32 * np.pi))) ** 2
            / (1024 * np.sqrt(2 * np.pi) * self.n0_norm)
        )

    n0 = 18 * np.pi
    second_moment_const = min(n0 - 16 * np.pi, (n0 + 16 * np.pi) ** 1.5 * (np.sqrt(2 * np.pi * (n0 + 16 * np.pi)) - 8 * np.pi) / (1024 * np.sqrt(2 * np.pi) * n0) * (np.log((n0 + 16 * np.pi) / (32 * np.pi))) ** 2)

    @staticmethod
    def second_moment_const_finder(n0):
        return (n0 + 16 * np.pi) ** 1.5 * (np.sqrt(2 * np.pi * (n0 + 16 * np.pi)) - 8 * np.pi) / (1024 * np.sqrt(2 * np.pi) * n0) * (np.log((n0 + 16 * np.pi) / (32 * np.pi))) ** 2 - (n0 - 16 * np.pi)

    def chi(self, x: np.ndarray) -> float:
        x = np.linalg.norm(x)
        return (np.cos(np.pi * 2 / self.a * x) + 1) / 2

    def u(self, x: np.ndarray) -> np.ndarray:
        return self.Q * (
                (x - self.a) / ((x - self.a) @ (x - self.a) + self.lem) * self.chi(x - self.a) +
                (x + self.a) / ((x + self.a) @ (x + self.a) + self.lem) * self.chi(x + self.a)
        )

    def phi(self, x: np.ndarray) -> float:
        """
        phi(x) = C0 * (h^2 - x^2)^3
        Integral(phi(x)|x|^2) < C1
        Thus 0 < C0 < 20*C1/(h^10*pi)
        We need Integral(phi(x)) = 9*pi
        Thus 0 < h < np.sqrt(5*C1)/(3*np.sqrt(pi))
        """
        ...
