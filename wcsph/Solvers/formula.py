import numpy as np


class WeaklyCompressibleSPH:
    def __init__(self):
        """
        Solver parameters
        """
        self.r = 0.01
        self.d = 2*self.r
        self.h = 0.05

        self.fluid_mass = 1.0
        self.boundary_mass = 10.0
        self.air_mass = 0.1

        self.fluid_particles = []
        self.air_particles = []
        self.boundary_particles = []

    def density(self, ) -> float:
        same_type_particle_number = 10
        different_type_particle_number = 10
        boundary_type_particle_number = 10
        rho = 0.0
        kernel_sum = 0.0
        for j in range(same_type_particle_number):
            kernel_value = self.kernel(self.h)
            rho += self.fluid_mass * kernel_value
            kernel_sum += self.fluid_particles[j].mass / self.fluid_particles[j].rho * kernel_value
        if kernel_sum < 1.0:
            # filter
            rho /= kernel_sum

        for j in range(different_type_particle_number):
            air_kernel_value = self.kernel(self.d)
            rho += self.air_mass * air_kernel_value
            ...

        boundary_kernel_sum = 0.0
        for j in range(boundary_type_particle_number):
            boundary_kernel_value = self.kernel(self.d)
            rho += self.boundary_mass * boundary_kernel_value
            ...

        ...


