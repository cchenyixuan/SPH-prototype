import numpy as np
from multiprocessing import Pool

r"""
The analytical solution is given by: 
%
\begin{equation}
    u(x, t) = \int G_t(x-y)f(y)dy
\end{equation}
%
where $G_t(x)$ is defined as:
%
\begin{equation}
    G_t(x) = \frac{1}{4\pi t}e^{-\frac{|x|^2}{4t}}
\end{equation}
%
"""

n0 = np.load(r"D:\ProgramFiles\PycharmProject\SPH-prototype\p_buffer6x6_for_acc.npy").reshape((-1, 4, 4))


def G(x: np.ndarray, t: float):
    if t == 0.0:
        return 0.0
    return 1/(4*np.pi*t)*np.exp(-(x@x)/(4*t))


def n(x, t):
    ans = 0.0
    for vertex in n0:
        y = np.array([vertex[0, 0], vertex[0, 2]])
        ans += vertex[1, 0]*G(x-y, t)*0.004*0.004
    print(t)
    return ans, t


def worker(args):
    return n(*args)


if __name__ == "__main__":
    pool = Pool(45)
    ans = pool.map(worker, [(np.array([0.0, 0.0]), 0.000005*i) for i in range(1, int(1.0//0.000005), 200)])
    with open("mp_ans00_wend_acc1.0.txt", "w") as f:
        for item in ans:
            f.write(f"{item[0]} {item[1]}\n")
        f.close()
    # for time_step in range(1000):
    #     print(n(np.array([0.0, 1.0]), 0.000005*(time_step+1)), 0.000005*(time_step+1))
