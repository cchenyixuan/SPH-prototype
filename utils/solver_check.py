import numpy as np
import matplotlib.pyplot as plt


class SolverCheck3D:
    def __init__(self, h, r, particle_buffer=None):
        self.H = h
        self.R = r

        # create a pcd around origin with x,y,z = +-h
        # regular distribution
        if particle_buffer is None:
            self.buffer = self.regular_distribution()
            self.particle_volume = 4 / 3 * np.pi * self.R ** 3
            self.particle_volume = 1 / self.kernel_sum("wendland_3d_c")
            self.kernel_sum("poly6_3d")
            self.kernel_sum("spiky_3d")
            self.kernel_sum("wendland_3d")
            self.kernel_sum("wendland_3d_c")
            self.grad_kernel_sum("grad_wendland_3d_c")
            self.buffer = self.irregular_distribution()
            self.kernel_sum("poly6_3d")
            self.kernel_sum("spiky_3d")
            self.kernel_sum("wendland_3d")
            self.kernel_sum("wendland_3d_c")
            self.grad_kernel_sum("grad_wendland_3d_c")
        else:
            self.buffer = particle_buffer
            self.particle_volume = 4 / 3 * np.pi * self.R ** 3
            self.particle_volume = 1 / self.kernel_sum("wendland_3d_c")
            self.kernel_sum("poly6_3d")
            self.kernel_sum("spiky_3d")
            self.kernel_sum("wendland_3d")
            self.kernel_sum("wendland_3d_c")
            self.grad_kernel_sum("grad_wendland_3d_c")

    def __call__(self, *args, **kwargs):
        return self.particle_volume

    def kernel_sum(self, kernel_name="poly6_3d"):
        kernels = {"poly6_3d": self.poly6_3d,
                   "spiky_3d": self.spiky_3d,
                   "wendland_3d": self.wendland_3d,
                   "wendland_3d_c": self.wendland_3d_c,
                   }
        kernel = kernels[kernel_name]
        ans, count = 0.0, 0.0
        for particle in self.buffer:
            kernel_tmp = kernel(np.linalg.norm(particle[:3]), self.H)
            ans += kernel_tmp
            count += bool(kernel_tmp)
        print(
            f"Kernel: {kernel_name}, Kernel Sum: {ans}, Particle Count: {count}, Kernel Integral: {ans * self.particle_volume}")
        return ans

    def grad_kernel_sum(self, kernel_name="grad_wendland_3d_c"):
        kernels = {
            "grad_wendland_3d_c": self.grad_wendland_3d_c,
        }
        kernel = kernels[kernel_name]
        ans, count = np.zeros((3,), dtype=np.longfloat), 0
        for particle in self.buffer:
            kernel_tmp = kernel(*particle[:3], np.linalg.norm(particle[:3]), self.H)
            ans += kernel_tmp
            count += bool(np.linalg.norm(kernel_tmp))
        print(
            f"Kernel: {kernel_name}, Kernel Sum: {ans}, Particle Count: {count}, Kernel Integral: {ans * self.particle_volume}")
        return ans

    @staticmethod
    def poly6_3d(rij, h):
        return max(0.0, 315 / (64 * np.pi * pow(h, 9)) * pow((h ** 2 - rij * rij), 3))

    @staticmethod
    def spiky_3d(rij, h):
        return max(0.0, 15 / (np.pi * pow(h, 6)) * pow((h - rij), 3))

    @staticmethod
    def wendland_3d(rij, h_2):
        h = h_2 * 0.5
        q = rij / h
        if q > 2:
            return 0.0
        return 495 / 256 / np.pi / h / h / h * pow(1 - q / 2, 6) * (35 / 12 * q * q + 3 * q + 1)

    @staticmethod
    def wendland_3d_c(rij, h):
        q = rij / h
        if q > 1:
            return 0.0
        return 495 / (32 * np.pi * h ** 3) * pow(1 - q, 6) * (35 / 3 * q * q + 6 * q + 1)

    @staticmethod
    def grad_wendland_3d_c(x, y, z, rij, h):
        q = rij / h
        if q > 1:
            return np.array([0.0, 0.0, 0.0], dtype=np.longfloat)
        if q == 0.0:
            return np.array([0.0, 0.0, 0.0], dtype=np.longfloat)
        w_prime = 495 / (32 * np.pi * h ** 3) / h * (-56 / 3) * q * (1 + 5 * q) * pow(1 - q, 5)
        return w_prime * np.array([x / rij, y / rij, z / rij], dtype=np.longfloat)

    def regular_distribution(self):
        origin = np.array([0.0, 0.0, 0.0], dtype=np.longfloat)

        shape_x = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        row = np.zeros((shape_x, 3), dtype=np.longfloat)
        offset_x = np.array([self.R * 2, 0.0, 0.0], dtype=np.longfloat)
        row[0] = origin
        for i in range(1, shape_x):
            row[i] = origin + offset_x * np.array([np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0, 0.0],
                                                  dtype=np.longfloat)

        shape_y = int(5 + 2 * ((self.H - self.R) // (np.sqrt(3) * self.R)))
        layer = np.zeros((shape_y, shape_x, 3), dtype=np.longfloat)
        offset_y = np.array([self.R, np.sqrt(3) * self.R, 0.0], dtype=np.longfloat)
        layer[0] = row
        for i in range(1, shape_y):
            layer[i] = row + offset_y * np.array([bool((i % 4 % 3)), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0],
                                                 dtype=np.longfloat)

        shape_z = int(5 + 2 * ((self.H - self.R) // (2 * np.sqrt(6) / 3 * self.R)))
        buffer = np.zeros((shape_z, shape_y, shape_x, 3), dtype=np.longfloat)
        offset_z = np.array([0.0, 2 / np.sqrt(3) * self.R, 2 * np.sqrt(6) / 3 * self.R], dtype=np.longfloat)
        buffer[0] = layer
        for i in range(1, shape_z):
            buffer[i] = layer + offset_z * np.array([0.0, bool((i % 4 % 3)), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1)],
                                                    dtype=np.longfloat)

        return buffer.reshape((-1, 3))

    def irregular_distribution(self):
        return np.random.random(self.buffer.shape) * self.R * 2 + self.buffer


class SolverCheck2D:
    def __init__(self, h, r):
        self.H = h
        self.R = r

        # create a pcd around origin with x,y,z = +-h
        # regular distribution
        self.buffer = self.star_distribution()
        # self.buffer = self.irregular_distribution()
        self.particle_volume = np.pi * self.R ** 2
        self.kernel_sum("wendland_2d")
        self.tv = 4 / 3 * np.pi * self.R ** 3
        self.particle_volume = 1 / self.kernel_sum("wendland_2d")
        print(self.particle_volume-self.tv)
        self.kernel_sum("poly6_2d")
        self.kernel_sum("spiky_2d")
        self.kernel_sum("wendland_2d")
        self.grad_kernel_sum("grad_wendland_2d")
        self.grad_kernel_sum("grad_spiky_2d")
        self.laplacian("grad_wendland_2d")
        self.laplacian("grad_spiky_2d")
        # self.buffer = self.irregular_distribution()
        self.particle_volume = 1 / self.kernel_sum("wendland_2d")
        self.kernel_sum("spiky_2d")
        self.kernel_sum("poly6_2d")
        self.kernel_sum("wendland_2d")
        self.grad_kernel_sum("grad_wendland_2d")
        self.grad_kernel_sum("grad_spiky_2d")
        self.laplacian("grad_wendland_2d")
        self.laplacian("grad_spiky_2d")
        self.outer_kernel_sum("grad_wendland_2d")

    def __call__(self, *args, **kwargs):
        return self.particle_volume

    def kernel_sum(self, kernel="poly6_2d"):
        kernels = {"poly6_2d": self.poly6_2d,
                   "spiky_2d": self.spiky_2d,
                   "wendland_2d": self.spiky_2d, }
        kernel = kernels[kernel]
        ans, count = 0.0, 0.0
        for particle in self.buffer:
            rij = np.linalg.norm(particle[:2])
            if rij != 0.0:
                kernel_tmp = kernel(np.linalg.norm(particle), self.H)
                ans += kernel_tmp
                count += bool(kernel_tmp)
        print(f"Kernel Sum: {ans}, Particle Count: {count}, Kernel Integral: {ans * self.particle_volume}")
        return ans

    def grad_kernel_sum(self, kernel_name="grad_wendland_2d"):
        kernels = {
            "grad_wendland_2d": self.grad_wendland_2d,
            "grad_spiky_2d": self.grad_spiky_2d,
        }
        kernel = kernels[kernel_name]
        ans, count = np.zeros((2,), dtype=np.longfloat), 0
        for particle in self.buffer:
            rij = np.linalg.norm(particle[:2])
            if rij != 0.0:

                kernel_tmp = kernel(*particle[:2], np.linalg.norm(particle[:2]), self.H)
                ans += kernel_tmp
                count += bool(np.linalg.norm(kernel_tmp))
        print(
            f"Kernel: {kernel_name}, Kernel Sum: {ans}, Particle Count: {count}, Kernel Integral: {ans * self.particle_volume}")
        return ans

    def outer_kernel_sum(self, kernel_name="grad_wendland_2d"):
        kernels = {
            "grad_wendland_2d": self.grad_wendland_2d,
            "grad_spiky_2d": self.grad_spiky_2d,
        }
        kernel = kernels[kernel_name]
        ans, count = np.zeros((2, 2), dtype=np.longfloat), 0
        for particle in self.buffer:
            rij = np.linalg.norm(particle[:2])
            if rij != 0.0:
                if True:
                    kernel_tmp = kernel(*particle[:2], np.linalg.norm(particle[:2]), self.H)
                    ans -= np.outer(particle[:2], kernel_tmp)
                    count += bool(np.linalg.norm(kernel_tmp))
        print(
            f"Kernel: {kernel_name}, Kernel Sum: {ans}, \nParticle Count: {count}, Outer Integral: {ans * self.particle_volume}")
        return ans

    def laplacian(self, kernel_name="grad_wendland_2d"):
        # Particle[particle_index-1][2].x -= particle_volume * (Particle[particle_index-1][2].z-Particle[index_j-1][2].z) * 2 * length(kernel_tmp)/(rij);
        kernels = {
            "grad_wendland_2d": self.grad_wendland_2d,
            "grad_spiky_2d": self.grad_spiky_2d,
        }
        kernel = kernels[kernel_name]
        ans, count = 0.0, 0
        for particle in self.buffer:
            rij = np.linalg.norm(particle[:2])
            if rij != 0.0:
                kernel_tmp = kernel(*particle[:2], rij, self.H)
                ans += np.linalg.norm(kernel_tmp) * 2 / rij
                count += bool(np.linalg.norm(kernel_tmp))
        print(
            f"Kernel: {kernel_name}, Laplacian: {ans * self.particle_volume}, Particle Count: {count}")
        return ans

    @staticmethod
    def poly6_2d(rij, h):
        return max(0.0, 4 / (np.pi * pow(h, 8)) * pow((h * h - rij * rij), 3))

    @staticmethod
    def spiky_2d(rij, h):
        return max(0.0, 10 / (np.pi * pow(h, 5)) * pow((h - rij), 3))

    @staticmethod
    def grad_spiky_2d(x, y, rij, h):
        if rij == 0:
            return np.array([0.0, 0.0], dtype=np.longfloat)
        if rij >= h:
            return np.array([0.0, 0.0], dtype=np.longfloat)
        else:
            w_prime = - 3 * 10 / (np.pi * pow(h, 5)) * pow((h - rij), 2)
            return np.array([w_prime * x / rij, w_prime * y / rij], dtype=np.longfloat)

    @staticmethod
    def wendland_2d(rij, h):
        coefficient = 9 / np.pi / h ** 2
        return coefficient * (1 - rij / h) ** 6 * (35 / 3 * rij / h * rij / h + 6 * rij / h + 1)

    @staticmethod
    def grad_wendland_2d(x, y, rij, h):
        if rij == 0.0:
            return np.zeros((2,), dtype=np.longfloat)
        if rij > h:
            return np.zeros((2,), dtype=np.longfloat)
        coefficient = 9 / np.pi / h ** 2
        return coefficient * (-56 / 3) * (1 / h / h) * (1 - rij / h) ** 5 * (5 * rij / h + 1) * np.array([x, y], dtype=np.longfloat)

    def star_distribution(self):
        origin = np.array([0.0, 0.0, 0.0], dtype=np.longfloat)
        shape_x = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        row = np.zeros((shape_x, 3), dtype=np.longfloat)
        offset_x = np.array([self.R * 2, 0.0, 0.0], dtype=np.longfloat)
        row[0] = origin
        for i in range(1, shape_x):
            row[i] = origin + offset_x * np.array([np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0, 0.0],
                                                  dtype=np.longfloat)
        shape_y = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        layer = np.zeros((shape_y, shape_x, 3), dtype=np.longfloat)
        offset_y = np.array([self.R, np.sqrt(3) * self.R, 0.0], dtype=np.longfloat)
        layer[0] = row
        for i in range(1, shape_y):
            layer[i] = row + offset_y * np.array(
                [np.sign((i - 1) % 4 - 1.5) * (((i - 1) // 2 + 1) % 2), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0],
                dtype=np.longfloat)

        return layer.reshape((-1, 3))

    def regular_distribution(self):
        origin = np.array([0.0, 0.0, 0.0], dtype=np.longfloat)

        shape_x = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        row = np.zeros((shape_x, 3), dtype=np.longfloat)
        offset_x = np.array([self.R * 2, 0.0, 0.0], dtype=np.longfloat)
        row[0] = origin
        for i in range(1, shape_x):
            row[i] = origin + offset_x * np.array([np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0, 0.0],
                                                  dtype=np.longfloat)

        shape_y = int(5 + 2 * ((self.H - self.R) // (2 * self.R)))
        layer = np.zeros((shape_y, shape_x, 3), dtype=np.longfloat)
        offset_y = np.array([self.R, 2 * self.R, 0.0], dtype=np.longfloat)
        layer[0] = row
        for i in range(1, shape_y):
            layer[i] = row + offset_y * np.array([0.0, np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1), 0.0],
                                                 dtype=np.longfloat)

        shape_z = 1  # int(5 + 2 * ((self.H - self.R) // (2 * np.sqrt(6) / 3 * self.R)))
        buffer = np.zeros((shape_z, shape_y, shape_x, 3), dtype=np.longfloat)
        offset_z = np.array([0.0, 2 / np.sqrt(3) * self.R, 2 * np.sqrt(6) / 3 * self.R], dtype=np.longfloat)
        buffer[0] = layer
        # for i in range(1, shape_z):
        #     buffer[i] = layer + offset_z * np.array([0.0, bool((i % 4 % 3)), np.sign(i % 2 - 0.5) * ((i - 1) // 2 + 1)],
        #                                             dtype=np.longfloat)

        return buffer.reshape((-1, 3))

    def irregular_distribution(self):
        return np.random.random(self.buffer.shape) * self.R * 1 + self.buffer


if __name__ == "__main__":
    h = 0.002
    r = 0.00005
    sc = SolverCheck2D(h, r)
    particle_volume = sc()
    print(particle_volume)
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ax.scatter(sc.buffer[:, 0], sc.buffer[:, 1], [sum(sc.grad_wendland_2d(p[0], p[1], np.linalg.norm(p[:2]), h)) for p in sc.buffer])
    # plt.xlim([-0.2 * h, 0.2 * h])
    # plt.ylim([-0.2 * h, 0.2 * h])
    plt.show()
