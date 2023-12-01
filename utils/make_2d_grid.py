import numpy as np


class Grid2D:
    """
    Center at origin.
    mat4{
        x        , y        , 0.0      , voxel_id ;
        grad(n).x, grad(n).y, grad(v).x, grad(v).y;
        lap(n)   , lap(v)   , n        , v        ;
        dn/dt    , 0.0      , u.x      , u.y      ;
    }
    """
    def __init__(self, h, r, x_range, y_range):
        self.H = h
        self.R = r
        self.x_num = int(x_range // (2 * self.R) * 2 + 1)
        self.y_num = int(y_range // (2 * self.R) * 2 + 1)
        self.buffer = np.zeros((self.x_num * self.y_num, 4, 4), dtype=np.float32)
        for i in range(self.buffer.shape[0]):
            self.buffer[i][0, 0], self.buffer[i][0, 2] = [(i//self.y_num - self.x_num//2) * self.R*2, (i % self.y_num - self.y_num//2) * self.R*2]
        print(f"Empty Grid2D Created with RangeX: [{-x_range}, {x_range}] RangeY: [{-y_range}, {y_range}] TotalNodes: {self.x_num * self.y_num} = {self.x_num}x{self.y_num}")

    def calculate_u(self):
        def chi(x: np.ndarray, a: float, ratio=0.5) -> float:
            """
            ratio controls where the cut-off occurs, e.g. 0.0: beginning, 1.0: no-cutoff
            """
            x = np.linalg.norm(x)
            if 0 <= x < ratio * a:
                return 1.0
            elif ratio * a <= x < a:
                return (np.cos(np.pi / ((1 - ratio) * a) * (x - ratio * a)) + 1) / 2
            else:
                return 0.0

        def u(x: np.ndarray, a=10.0, ratio=0.5, Q=0.005, lam=1.0) -> np.ndarray:
            vector_a = np.array([a, 0.0], dtype=np.float32)
            return Q * ((x - vector_a) / ((x - vector_a) @ (x - vector_a).T + lam) * chi(x - vector_a, a, ratio) + (
                        x + vector_a) / ((x + vector_a) @ (x + vector_a).T + lam) * chi(x + vector_a, a, ratio))

        for i in range(self.buffer.shape[0]):
            self.buffer[i][3, 2:] = u(np.array([self.buffer[i][0, 0], self.buffer[i][0, 2]]))

        return self.buffer

    def calculate_n(self):
        def phi(x: np.ndarray, c0=1684.3988524066801, h=0.6183469108724111):
            return max(0.0, c0 * (h ** 2 - x @ x.T) ** 3)

        def n(x: np.ndarray, t=0.0, a=np.array([10.0, 0.0])):
            return phi(x - a) + phi(x + a)

        for i in range(self.buffer.shape[0]):
            self.buffer[i][2, 2] = n(np.array([self.buffer[i][0, 0], self.buffer[i][0, 2]]))

        return self.buffer

    def calculate_v(self):
        ...

    def __call__(self, function):

        ...


if __name__ == "__main__":
    test_grid = Grid2D(0.05, 0.005, 15, 5)
    test_grid.calculate_u()
    test_grid.calculate_n()
    np.save("../p_buffer15.npy", test_grid.buffer)

