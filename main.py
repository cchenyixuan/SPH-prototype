from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import math

from typing import Tuple

from kernels import Kernels

ndarray = np.ndarray

PI = math.pi
REST_DENS = 1000
EOS_CONST = 2000
H = 0.1
HSQ = H * H
MASS = 2
RADIUS = 0.05
VISC = 10
DT = 0.008
G = -9.81


def draw_particles(particles: ndarray, color):
    for i, data in enumerate(particles):
        x, y = data[0], data[1]
        circ = plt.Circle((x, y), RADIUS, color=color, alpha=0.3)
        ax.add_artist(circ)
        ax.set_aspect("equal")
        ax.set_xlim([0.0-RADIUS, 5.0+RADIUS])
        ax.set_ylim([0.0-4*RADIUS, 5.0+RADIUS])


class Particle:
    def __init__(self, mass: float, pos: ndarray, vel: ndarray, rho: float, press: float) -> None:
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.rho = rho
        self.press = press

    def __call__(self):
        return self.pos


class ParticleGroup:
    def __init__(self, center: Tuple[float, float], width: float, height: float, n_particles: int) -> None:
        """self.center = center
        self.width = width
        self.height = height
        self.left = center[0] - 0.5 * width
        self.right = center[0] + 0.5 * width
        self.top = center[1] + 0.5 * height
        self.bottom = center[1] - 0.5 * height

        self.n_particles = 40000  # n_particles  400*100
        self.particles = np.zeros((self.n_particles, 13),
                                  dtype=np.float32)  # this is a buffer of data containing (pos vel m r p i) * n
"""
        # mass, rho, press = 0.01, 1000, 0
        tmp = []
        for x in range(20):
            for y in range(20):
                eps_x = np.random.randn() / 200
                eps_y = np.random.randn() / 200
                tmp.append(np.array([1.0+RADIUS*2*x+eps_x, 2.0+RADIUS*2*y+eps_y, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.particles = np.array(tmp, dtype=np.float32)

        # self.hash_domain = ...
        tmp = []
        for x in range(40):
            for y in range(2):
                eps_x = np.random.randn() / 200
                eps_y = np.random.randn() / 200
                tmp.append(np.array(
                    [0.0 + RADIUS * 2 * x + eps_x, -0.15 + RADIUS * 2 * y + eps_y, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                     0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.ghost = np.array(tmp, dtype=np.float32)


class SPHSolver:
    def __init__(self, particles, boundary):
        self.particles = particles
        self.boundary = boundary
        self.all_particles = np.vstack((self.particles, self.boundary))

    def compute_2d_density_and_pressure(self):
        # self.particles[i] [0:3]pos [3:6]vel [6]mass [7]rho [8]press [9]id

        # all particles in domain
        for i, particle_i in enumerate(self.all_particles[:]):

            if i % 10 == 0:
                print(f"dense {i}")

            rho = 0.0
            pos_i = particle_i[:3]

            # all particles in domain
            for j, particle_j in enumerate(self.all_particles):

                pos_j = particle_j[:3]
                xij = pos_i-pos_j
                rij = np.linalg.norm(xij)
                if rij < H:
                    rho += MASS * Kernels.poly6_3d(xij, rij, H)


            # end loop

            particle_i[7] = rho
            particle_i[8] = EOS_CONST * (particle_i[7]/REST_DENS - 1)

    def compute_2d_external_force(self):
        # self.particles[i] [0:3]pos [3:6]vel [6]mass [7]rho [8]press [9]id [10:13]force
        print("before", self.all_particles[0])
        # all particles in domain
        for i, particle_i in enumerate(self.all_particles[:400]):
            if i % 100 == 0:
                print(f"force {i}")

            pos_i = particle_i[:3]

            f_press = np.array((0.0, 0.0, 0.0), dtype=np.float32)
            f_viscosity = np.array((0.0, 0.0, 0.0), dtype=np.float32)

            # all other particles in domain except current focus
            for j, particle_j in enumerate(self.all_particles):
                if i == j:
                    continue
                pos_j = particle_j[:3]
                xij = -pos_i + pos_j
                rij = np.linalg.norm(xij)
                if rij < H:
                    f_press += Kernels.grad_spiky_3d(xij, rij, H) * (MASS*(particle_j[8]/particle_j[7]**2 + particle_i[8]/particle_i[7]**2)) * particle_i[7]
                    # f_press += - Kernels.grad_spiky_2d(xij, rij, H) * MASS*(particle_j[8]+particle_i[8])/(2 * particle_i[7])
                    f_viscosity += VISC * (MASS / particle_j[7]) * (particle_j[3:6] - particle_i[3:6]) * Kernels.lap_viscosity_3d(xij, rij, H)

            f_gravity = np.array((0.0, G, 0.0), dtype=np.float32)*particle_i[7]
            particle_i[13:16] = f_press
            particle_i[16:19] = f_viscosity
            particle_i[10:13] = f_press + f_viscosity + f_gravity
        print("after", self.all_particles[0])
        # update
        for i, particle_i in enumerate(self.all_particles[:400]):
            particle_i[3:6] += DT * particle_i[10:13] / particle_i[7]
            particle_i[:3] += DT * particle_i[3:6]

            if particle_i[0] - RADIUS < -0.05:
                particle_i[3] *= -0.25
                particle_i[0] = -0.05 + RADIUS
            if particle_i[0] + RADIUS > 4.05:
                particle_i[3] *= -0.25
                particle_i[0] = 4.05 - RADIUS
            if particle_i[1] - RADIUS < -0.05:
                particle_i[4] *= -0.25
                particle_i[1] = -0.05 + RADIUS
            if particle_i[1] + RADIUS > 4.05:
                particle_i[4] *= -0.25
                particle_i[1] = 4.05 - RADIUS

    def update(self):
        self.compute_2d_density_and_pressure()
        self.compute_2d_external_force()


if __name__ == "__main__":
    import os
    os.makedirs("a", exist_ok=True)
    a = ParticleGroup((250, 250), 500, 500, 40000)
    solver = SPHSolver(a.particles, a.ghost)
    for i in range(5000):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        fig.set_tight_layout(True)

        solver.update()
        draw_particles(solver.all_particles[400:], "grey")
        draw_particles(solver.all_particles[:400], "blue")
        plt.savefig(f'a/c_damped_{i}.png')
        # plt.show()
        plt.close()
