import time

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def bessel_2d(x, gamma=1.0, alpha=2.0):
    x = np.linalg.norm(x)  # x = |x|
    coefficient = 1 / (2 * np.pi * np.sqrt(2))  # a_d

    return coefficient * gamma ** ((2 - alpha) / 2) * np.exp(-np.sqrt(gamma) * x) * integral(x, gamma=gamma, alpha=alpha,
                                                                                             ds=0.0001)


def norm_grad_bessel_2d(x: np.ndarray, gamma=1.0, alpha=2.0) -> float:
    x = np.linalg.norm(x)  # x = |x|
    coefficient = 1 / (2 * np.pi * np.sqrt(2))  # a_d

    return gamma ** ((2 - alpha + 1) / 2) * coefficient * np.exp(-np.sqrt(gamma) * x) * integral_grad(x, gamma, alpha,
                                                                                                      ds=0.0001)


def c_star(n_0, c_gamma=0.2, Q=0.005):
    return 4 * np.sqrt(2) * Q + 16 * (np.sqrt(2) + 6) + 4 * np.sqrt(2) * c_gamma * n_0 * \
        (
                8 * np.sqrt(2) * (2 + (n_0 + 16 * np.pi) * (np.sqrt(2 * np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi) / (
                128 * np.pi * c_gamma * n_0 ** 2)) +
                48 * (1 + (64 * np.pi * c_gamma * n_0 ** 2) / (n_0 + 16 * np.pi) / (
                np.sqrt(2 * np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi)) ** 2 +
                8 * np.sqrt(2) * (1 + (64 * np.pi * c_gamma * n_0 ** 2) / (n_0 + 16 * np.pi) / (
                np.sqrt(2 * np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi)) + 3

        )


def l(n_0, c_gamma=0.2, Q=0.005, lam=1.0, h=0.6183469108724111):
    c_s = c_star(n_0, c_gamma=c_gamma, Q=Q)
    return max(
        3 * h,
        32 * np.sqrt((6 * np.pi * n_0 * (2 + c_gamma * n_0)) / (
                    (n_0 + 16 * np.pi) * (np.sqrt(2 * np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi))),
        2 * h + (256 * np.pi * np.sqrt(3 * c_s * n_0 * (n_0 - 16 * np.pi) * (2 + c_gamma * n_0))) / (
                    (n_0 + 16 * np.pi) * (np.sqrt(2*np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi)),
        (1024 * np.sqrt(2) * np.pi * n_0 * Q * max(1.0, 1.0 / lam)) / (
                    (n_0 + 16 * np.pi) * (np.sqrt(2*np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi)),
        2 * h + (16384 * np.sqrt(2) * np.pi ** 2 * c_s * n_0 * (n_0 - 16 * np.pi) * Q * max(1.0, 1.0 / lam)) / (
                    (n_0 + 16 * np.pi) ** 2 * (np.sqrt(2*np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi) ** 2)
    )


def k_0(n_0, c_gamma=0.2):
    return 2*l(n_0, c_gamma=c_gamma)*(1+(68*np.pi*c_gamma*n_0**2)/((n_0+16*np.pi)*(np.sqrt(2*np.pi * (n_0 + 16 * np.pi)) - 8 * np.pi)))


def integral_grad(x, gamma=1.0, alpha=2.0, ds=0.0001):
    x = abs(x)
    ans = 0.0
    s_list = (0.0001 + _ * ds for _ in range(int(500 / ds)))
    for s in s_list:
        ans += (1 + s) * np.exp(-np.sqrt(gamma) * x * s) * (s + s * s / 2) ** ((2 - alpha - 1) / 2) * ds
    return ans


def poisson_2d(x: float):
    x = abs(x)
    if x == 0.0:
        return 0
    return -np.log(x) / (2 * np.pi)


def integral(x, gamma=1.0, alpha=2.0, ds=0.0001):
    x = abs(x)
    ans = 0.0
    s_additional = (0.000001 + _ * 0.000001 for _ in range(100))
    s_list = (0.0001 + _ * ds for _ in range(int(500 / ds)))
    for s in s_list:
        ans += np.exp(-np.sqrt(gamma) * x * s) * (s + s * s / 2) ** ((2 - alpha - 1) / 2) * ds
    return ans


# bessel_2d_data = np.load(r"KS/result2.npy")


def get_kernel_value(x, y):
    norm = np.linalg.norm([x, y])
    if norm > 9.999:
        return 0.0
    a = int(norm // 0.001)
    b = a + 1
    theta = norm % 0.001 / 0.001
    return bessel_2d_data[1, a] * (1 - theta) + bessel_2d_data[1, b] * theta


def discrete_kernel():
    kernel = np.zeros((1999, 1999), dtype=np.float32)
    quarter_kernel = np.zeros((999, 999), dtype=np.float32)
    for i in range(1, 1000):
        for j in range(1, 1000):
            quarter_kernel[i - 1, j - 1] = get_kernel_value(i * 0.01, j * 0.01)
    line_kernel = np.zeros((999,), dtype=np.float32)
    for i in range(1, 999):
        line_kernel[i - 1] = get_kernel_value(i * 0.01, 0.0)
    kernel[999, 1000:] = line_kernel
    kernel[999, :999] = line_kernel[::-1]
    kernel[:999, 999] = line_kernel[::-1]
    kernel[1000:, 999] = line_kernel

    kernel[1000:, 1000:] = quarter_kernel
    kernel[:999, 1000:] = quarter_kernel[::-1]
    kernel[:999, :999] = quarter_kernel[::-1, ::-1]
    kernel[1000:, :999] = quarter_kernel[:, ::-1]
    return kernel


def convolution_2d(matrix, kernel):
    # memory-consumption method
    pad = np.zeros((matrix.shape[0] + 2000, matrix.shape[1] + 2000), dtype=np.float32)
    pad[1000:1000 + matrix.shape[0], 1000:1000 + matrix.shape[1]] = matrix
    ans = np.zeros_like(matrix)

    def conv_worker(i):
        ans_ = []
        print(i)
        for j in range(1000, 1000 + matrix.shape[1]):
            ans_.append(np.sum(kernel * pad[i - 1000:i + 999, j - 1000:j + 999]))
        return ans_

    pool = Pool()
    ans__ = pool.map(conv_worker, range(1000, 1000 + matrix.shape[0]))
    # for i in range(1000, 1000+matrix.shape[0]):
    #     print(i)
    #     for j in range(1000, 1000+matrix.shape[1]):
    #         ans[i-1000, j-1000] = np.sum(kernel*pad[i-1000:i+999, j-1000:j+999])
    return np.array(ans__)


bessel_2d_data = np.load(r"D:\ProgramFiles\PycharmProject\SPH-prototype\KS/result2.npy")
kernel = discrete_kernel()
kernel = kernel[749:-749, 749:-749]
# u_map = np.load(r"D:\ProgramFiles\PycharmProject\SPH-prototype\p_buffer.npy")
u = np.zeros((2001, 2001), dtype=np.float32)


def kernel_phi(rij, h):
    return max(0.0, 4 / (np.pi * h ** 8) * (h ** 2 - rij * rij) ** 3)


for i in range(2001 * 2001):
    u[i // 2001, i % 2001] = 9 * np.pi * (
            kernel_phi(np.linalg.norm([((i // 2001) - 1000) * 0.01, ((i % 2001) - 1000 - 50) * 0.01]),
                       0.46845) + kernel_phi(
        np.linalg.norm([((i // 2001) - 1000) * 0.01, ((i % 2001) - 1000 + 50) * 0.01]), 0.46845))
pad = np.zeros((u.shape[0] + 250 * 2, u.shape[1] + 250 * 2), dtype=np.float32)
pad[250:250 + u.shape[0], 250:250 + u.shape[1]] = u


def conv_worker(i):
    print(i - 250)
    ans_ = []
    for j in range(250, 250 + 2001):
        res = kernel * pad[i - 250:i + 251, j - 250:j + 251]
        ans_.append(np.sum(res) * 0.0001)
    return ans_


if __name__ == "__main__":
    pool = Pool(20)

    t = time.time()
    ans__ = pool.map(conv_worker, [*range(250, 250 + 2001)])
    print(time.time() - t)
    np.save("vsmall500kernel.npy", np.array(ans__))
