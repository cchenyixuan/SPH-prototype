import numpy as np


class SolverCheck3D:
    def __init__(self, h, r):
        self.H = h
        self.R = r

        # create a pcd around origin with x,y,z = +-h
        # regular distribution
        self.buffer = self.regular_distribution()
        self.particle_volume = 4/3*np.pi*self.R**3
        self.particle_volume = 1/self.kernel_sum()
        self.kernel_sum("spiky_3d")
        self.buffer = self.irregular_distribution()
        self.kernel_sum("spiky_3d")
        self.kernel_sum()

    def __call__(self, *args, **kwargs):
        return self.particle_volume

    def kernel_sum(self, kernel="poly6_3d"):
        kernels = {"poly6_3d": self.poly6_3d,
                   "spiky_3d": self.spiky_3d, }
        kernel = kernels[kernel]
        ans, count = 0.0, 0.0
        for particle in self.buffer:
            kernel_tmp = kernel(np.linalg.norm(particle), self.H)
            ans += kernel_tmp
            count += bool(kernel_tmp)
        print(f"Kernel Sum: {ans}, Particle Count: {count}, Kernel Integral: {ans * self.particle_volume}")
        return ans

    @staticmethod
    def poly6_3d(rij, h):
        return max(0.0, 315 / (64 * np.pi * pow(h, 9)) * pow((h ** 2 - rij * rij), 3))

    @staticmethod
    def spiky_3d(rij, h):
        return max(0.0, 15 / (np.pi * pow(h, 6)) * pow((h - rij), 3))

    def regular_distribution(self):
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        shape_x = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        row = np.zeros((shape_x, 3), dtype=np.float64)
        offset_x = np.array([self.R * 2, 0.0, 0.0], dtype=np.float64)
        row[0] = origin
        for i in range(1, shape_x):
            row[i] = origin + offset_x * np.array([np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0, 0.0],
                                                  dtype=np.float64)

        shape_y = int(5 + 2 * ((self.H - self.R) // (np.sqrt(3) * self.R)))
        layer = np.zeros((shape_y, shape_x, 3), dtype=np.float64)
        offset_y = np.array([self.R, np.sqrt(3) * self.R, 0.0], dtype=np.float64)
        layer[0] = row
        for i in range(1, shape_y):
            layer[i] = row + offset_y * np.array([bool((i % 4 % 3)), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0],
                                                 dtype=np.float64)

        shape_z = int(5 + 2 * ((self.H - self.R) // (2 * np.sqrt(6) / 3 * self.R)))
        buffer = np.zeros((shape_z, shape_y, shape_x, 3), dtype=np.float64)
        offset_z = np.array([0.0, 2 / np.sqrt(3) * self.R, 2 * np.sqrt(6) / 3 * self.R], dtype=np.float64)
        buffer[0] = layer
        for i in range(1, shape_z):
            buffer[i] = layer + offset_z * np.array([0.0, bool((i % 4 % 3)), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1)],
                                                    dtype=np.float64)

        return buffer.reshape((-1, 3))

    def irregular_distribution(self):
        return np.random.random(self.buffer.shape)*self.R*2 + self.buffer


if __name__ == "__main__":
    sc = SolverCheck3D(0.01, 0.001)
    particle_volume = sc()
    print(particle_volume)

