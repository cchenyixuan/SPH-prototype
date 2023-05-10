from __future__ import annotations

import numpy as np
import math

ndarray = np.ndarray

PI = math.pi


class Kernels:
    def __init__(self):
        ...

    # Poly6
    @staticmethod
    def poly6_2d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, (4 * (h ** 2 - rij ** 2) ** 3) / (PI * h ** 8))

    @staticmethod
    def poly6_3d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, (315 * (h ** 2 - rij ** 2) ** 3) / (64 * PI * h ** 9))

    @staticmethod
    def grad_poly6_2d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        grad = np.array((0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = (-24 * (h ** 2 - rij ** 2) ** 2 * rij) / (PI * h ** 8)
        grad[0] = x / rij * w_prime
        grad[1] = y / rij * w_prime
        return grad

    @staticmethod
    def grad_poly6_3d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        z = xij[2]
        grad = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = (-945 * (h ** 2 - rij ** 2) ** 2 * rij) / (32 * PI * h ** 9)
        grad[0] = x / rij * w_prime
        grad[1] = y / rij * w_prime
        grad[2] = z / rij * w_prime
        return grad

    @staticmethod
    def lap_poly6_2d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0.0
        return (-48 * (h ** 2 - rij ** 2) * (h ** 2 - 3 * rij ** 2)) / (PI * h ** 8)

    @staticmethod
    def lap_poly6_3d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0.0
        return (-945 * (h ** 2 - rij ** 2) * (3 * h ** 2 - 7 * rij ** 2)) / (32 * PI * h ** 9)

    # Spiky
    @staticmethod
    def spiky_2d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, 10 / (PI * h ** 5) * (h - rij) ** 3)

    @staticmethod
    def spiky_3d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, 15 / (PI * h ** 6) * (h - rij) ** 3)

    @staticmethod
    def grad_spiky_2d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        grad = np.array((0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = (-30 * (h - rij) ** 2) / (PI * h ** 5)
        grad[0] = x / rij * w_prime  # TODO: avoid nan, buggy
        grad[1] = y / rij * w_prime  # TODO: avoid nan, buggy
        return grad

    @staticmethod
    def grad_spiky_3d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        z = xij[2]
        grad = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = (-45 * (h - rij) ** 2) / (PI * h ** 6)
        grad[0] = x / rij * w_prime
        grad[1] = y / rij * w_prime
        grad[2] = z / rij * w_prime
        return grad

    @staticmethod
    def lap_spiky_2d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0.0
        return -6*h**2/rij + 18*h - 12*rij

    @staticmethod
    def lap_spiky_3d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0.0
        return -3*h**2/rij + 12*h - 9*rij

    # Viscosity
    @staticmethod
    def viscosity_2d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, (40 / (PI * h ** 2)) * (rij**2/(4*h**2) - rij**3/(9*h**3) - np.log(rij/h)/6 - 5/36))

    @staticmethod
    def viscosity_3d(xij: ndarray | None, rij: float, h: float) -> float:
        return max(0.0, (15 / (2*PI * h ** 3)) * (-rij**3/(2*h**3) + rij**2/h**2 + h/(2*rij) - 1))

    @staticmethod
    def grad_viscosity_2d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        grad = np.array((0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = 40*(rij/(2*h**2) - rij**2/(3*h**3) - 1/(6*rij)) / (PI*h**2)
        grad[0] = x / rij * w_prime
        grad[1] = y / rij * w_prime
        return grad

    @staticmethod
    def grad_viscosity_3d(xij: ndarray | None, rij: float, h: float) -> ndarray:
        x = xij[0]
        y = xij[1]
        z = xij[2]
        grad = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        if rij > h:
            return grad
        w_prime = 15 * (-3*rij**2/(2*h**3) + 2*rij/(h**2) - h/(2*rij**2)) / (2*PI * h ** 3)
        grad[0] = x / rij * w_prime
        grad[1] = y / rij * w_prime
        grad[2] = z / rij * w_prime
        return grad

    @staticmethod
    def lap_viscosity_2d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0
        return 40*(h-rij) / (PI*h**5)

    @staticmethod
    def lap_viscosity_3d(xij: ndarray | None, rij: float, h: float) -> float:
        if rij > h:
            return 0
        return 45*(h-rij) / (PI*h**6)

















