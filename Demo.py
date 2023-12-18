import numpy as np

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import time
from utils.shader_auto_completion import complete_shader
from utils.solver_check import SolverCheck3D

from project_loader import Project


class Demo:
    def __init__(self):
        # --case parameters--
        self.H = 0.05
        self.R = 0.005
        self.c0 = 60  # about 10 times of max velocity in this system ensures the density variation less than 1%
        self.DELTA_T = 0.4*self.H/self.c0  # check CFL condition
        self.save_frequency = int((1/self.DELTA_T)/100)  # approx. 0.01s
        self.VISCOSITY = 0.004
        self.COHESION = 0.001
        self.ADHESION = 0.0
        self.REST_DENSE = 1000.0
        self.EOS_CONSTANT = self.c0**2 * self.REST_DENSE / 7  # c0**2*rho0/gamma
        self.PARTICLE_VOLUME = SolverCheck3D(self.H, self.R)()

        self.voxel_buffer_file = r"D:\ProgramFiles\PycharmProject\VoxelizationAlg\voxelization\buffer.npy"
        self.voxel_origin_offset = [-1.3023989, -1.304125,  -1.306157]
        self.domain_particle_file = r".\models\domain.obj"
        self.boundary_particle_file = r"models\demo_boundary.obj"

        self.INLET1_file = r"models\demo_INLET1.obj"
        self.INLET1_velocity = [0.0, 0.0, -0.5]
        self.INLET1_area = 0.075**2*np.pi
        self.INLET2_file = r"models\demo_INLET2.obj"
        self.INLET2_velocity = [-0.5, 0.0, 0.0]
        self.INLET2_area = 0.075**2*np.pi
        self.INLET3_file = r"models\demo_INLET1.obj"
        self.INLET3_velocity = [0.0, 0.0, 0.0]
        self.INLET3_area = 0.00275**2*np.pi

        self.INLET1_FLUX = self.INLET1_area * np.linalg.norm(self.INLET1_velocity)  # S1*||Velocity1||
        self.INLET2_FLUX = self.INLET2_area * np.linalg.norm(self.INLET2_velocity)
        self.INLET3_FLUX = self.INLET3_area * np.linalg.norm(self.INLET3_velocity)
        # --solver parameters--
        self.VOXEL_MEMORY_LENGTH = 2912  # (2+60+60+60)*16
        self.VOXEL_BLOCK_SIZE = 960  # 60*16
        self.VOXEL_GROUP_SIZE = 300000  # one ssbo could get unstable if size is larger than (2912*4bit)*300000
        self.VOXEL_MATRIX_NUMBER = 182  # 2+60+60+60
        self.MAX_PARTICLE_NUMBER = 5000000

        self.project = Project(
            self.H,
            self.R,
            self.PARTICLE_VOLUME,
            self.REST_DENSE,
            self.voxel_buffer_file,
            self.voxel_origin_offset,
            self.domain_particle_file,
            self.boundary_particle_file,
            [(self.INLET1_file, self.INLET1_velocity), (self.INLET2_file, self.INLET2_velocity), (self.INLET3_file, self.INLET3_velocity)])
        self.voxels = self.project.voxels
        self.offset = self.project.offset  # voxel offset
        self.particles = self.project.particles
        self.boundary_particles = self.project.boundary_particles
        self.inlet1_particles, self.inlet2_particles, self.inlet3_particles = self.project.inlet_particles  # 3

        self.inlet_particle_number = self.inlet1_particles.shape[0]//4 + self.inlet2_particles.shape[0]//4 + self.inlet3_particles.shape[0]//4
        # particle_sub_data_buffer
        self.particles_sub_data = self.project.particles_buffer

        self.voxel_number = self.voxels.shape[0] // (self.VOXEL_MATRIX_NUMBER*4)  # (n * (182*4), 4)
        self.voxel_groups = [self.voxels[self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*i:self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*(i+1)] for i in range(self.voxel_number//self.VOXEL_GROUP_SIZE)]
        if self.voxel_number % self.VOXEL_GROUP_SIZE != 0:
            self.voxel_groups.append(self.voxels[self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*(self.voxel_number//self.VOXEL_GROUP_SIZE):])
        print(f"{len(self.voxel_groups)} voxel groups created.")

        self.particle_number = self.particles.shape[0] // 4  # (n * 4, 4)
        self.boundary_particle_number = self.boundary_particles.shape[0] // 4

        self.voxel_particle_numbers = np.zeros((self.voxel_number, ), dtype=np.int32)

        self.indices_buffer = np.array([i for i in range(max(self.particle_number, self.boundary_particle_number, self.voxel_number, self.MAX_PARTICLE_NUMBER))], dtype=np.int32)

        print(self.particle_number, self.boundary_particle_number, self.voxel_number)
        # global status buffer
        # [n_particle, ptr_last_particle, n_voxel, voxel_memory_length, voxel_block_size, voxel_group_size]
        self.global_status = np.array((self.particle_number, self.particle_number, self.voxel_number, self.inlet1_particles.shape[0]//4, self.inlet2_particles.shape[0]//4, self.inlet3_particles.shape[0]//4, 0, 0, 0, 0, 0, 0), dtype=np.int32)
        self.global_status_float = np.array((self.H, self.R, self.DELTA_T, self.VISCOSITY, self.COHESION, self.ADHESION, *self.offset, 0.0, 0.0, 0.0), dtype=np.float32)

        # all shaders are filled with certain parameters and saved in ./runtime
        complete_shader({
                "PI": np.pi,
                "n_boundary_particle": self.boundary_particle_number,
                "n_voxel": self.voxel_number,
                "h": self.H,
                "r": self.R,
                "voxel_memory_length": self.VOXEL_MEMORY_LENGTH,
                "voxel_block_size": self.VOXEL_BLOCK_SIZE,
                "rest_dense": self.REST_DENSE,
                "eos_constant": self.EOS_CONSTANT,
                "delta_t": self.DELTA_T,
                "viscosity": self.VISCOSITY,
                "cohesion": self.COHESION,
                "adhesion": self.ADHESION,
                "offset": self.offset,
                "VOXEL_GROUP_SIZE": self.VOXEL_GROUP_SIZE,
                "particle_volume": self.PARTICLE_VOLUME,
                "Coeff_Poly6_2d": 4 / (np.pi * self.H ** 8),
                "Coeff_Poly6_3d": 315 / (64 * np.pi * self.H ** 9),
                "Coeff_Spiky_2d": 10 / (np.pi * self.H ** 5),
                "Coeff_Spiky_3d": 15 / (np.pi * self.H ** 6),
                "Coeff_Viscosity_2d": 40 / (np.pi * self.H ** 2),
                "Coeff_Viscosity_3d": 15 / (2 * np.pi * self.H ** 3),
                "Coeff_Wendland_3d": 495 / (32 * np.pi * self.H ** 3),
                "MAX_PARTICLE_MASS": 6*self.REST_DENSE*self.PARTICLE_VOLUME,
                "ORIGINAL_PARTICLE_MASS": self.REST_DENSE*self.PARTICLE_VOLUME,
        })
        # initialize OpenGL
        # particles buffer
        self.sbo_particles = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.sbo_particles)
        glNamedBufferStorage(self.sbo_particles, np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32).nbytes, np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32), GL_DYNAMIC_STORAGE_BIT)
        glNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, self.particles)

        # particles sub data buffer
        self.sbo_particles_sub_data = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_particles_sub_data)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.sbo_particles_sub_data)
        glNamedBufferStorage(self.sbo_particles_sub_data, np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32).nbytes, np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32), GL_DYNAMIC_STORAGE_BIT)
        glNamedBufferSubData(self.sbo_particles_sub_data, 0, self.particles_sub_data.nbytes, self.particles_sub_data)

        # boundary buffer
        self.sbo_boundary_particles = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_boundary_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.sbo_boundary_particles)
        glNamedBufferStorage(self.sbo_boundary_particles, self.boundary_particles.nbytes, self.boundary_particles,
                             GL_DYNAMIC_STORAGE_BIT)
        # voxels buffer
        for index, buffer in enumerate(self.voxel_groups):
            self.__setattr__(f"sbo_voxels_{index}", glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.__getattribute__(f"sbo_voxels_{index}"))
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10+index, self.__getattribute__(f"sbo_voxels_{index}"))
            glNamedBufferStorage(self.__getattribute__(f"sbo_voxels_{index}"), buffer.nbytes, buffer, GL_DYNAMIC_STORAGE_BIT)

        # voxel_particle_numbers buffer
        self.sbo_voxel_particle_numbers = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_voxel_particle_numbers)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.sbo_voxel_particle_numbers)
        glNamedBufferStorage(self.sbo_voxel_particle_numbers, self.voxel_particle_numbers.nbytes, self.voxel_particle_numbers, GL_DYNAMIC_STORAGE_BIT)

        # voxel_particle_in_numbers buffer
        self.sbo_voxel_particle_in_numbers = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_voxel_particle_in_numbers)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.sbo_voxel_particle_in_numbers)
        glNamedBufferStorage(self.sbo_voxel_particle_in_numbers, self.voxel_particle_numbers.nbytes, self.voxel_particle_numbers, GL_DYNAMIC_STORAGE_BIT)

        # voxel_particle_out_numbers buffer
        self.sbo_voxel_particle_out_numbers = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_voxel_particle_out_numbers)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.sbo_voxel_particle_out_numbers)
        glNamedBufferStorage(self.sbo_voxel_particle_out_numbers, self.voxel_particle_numbers.nbytes, self.voxel_particle_numbers, GL_DYNAMIC_STORAGE_BIT)

        # global_status buffer
        self.sbo_global_status = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_global_status)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.sbo_global_status)
        glNamedBufferStorage(self.sbo_global_status, self.global_status.nbytes, self.global_status, GL_DYNAMIC_STORAGE_BIT)

        # global_status2 buffer
        self.sbo_global_status2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_global_status2)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, self.sbo_global_status2)
        glNamedBufferStorage(self.sbo_global_status2, self.global_status_float.nbytes, self.global_status_float, GL_DYNAMIC_STORAGE_BIT)

        # inlet1 buffer
        if self.inlet1_particles.nbytes == 0:
            self.inlet1_particles = np.zeros((4, 4), dtype=np.float32)
        self.sbo_inlet1 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_inlet1)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self.sbo_inlet1)
        glNamedBufferStorage(self.sbo_inlet1, self.inlet1_particles.nbytes, self.inlet1_particles, GL_DYNAMIC_STORAGE_BIT)

        # inlet2 buffer
        if self.inlet2_particles.nbytes == 0:
            self.inlet2_particles = np.zeros((4, 4), dtype=np.float32)
        self.sbo_inlet2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_inlet2)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, self.sbo_inlet2)
        glNamedBufferStorage(self.sbo_inlet2, self.inlet2_particles.nbytes, self.inlet2_particles,
                             GL_DYNAMIC_STORAGE_BIT)

        # inlet3 buffer
        if self.inlet3_particles.nbytes == 0:
            self.inlet3_particles = np.zeros((4, 4), dtype=np.float32)
        self.sbo_inlet3 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_inlet3)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, self.sbo_inlet3)
        glNamedBufferStorage(self.sbo_inlet3, self.inlet3_particles.nbytes, self.inlet3_particles, GL_DYNAMIC_STORAGE_BIT)

        # vao of indices
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.indices_buffer.nbytes, self.indices_buffer, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribIPointer(0, 1, GL_INT, 4, ctypes.c_void_p(0))

        # compute shader
        self.need_init = True

        # compute shader 0
        self.compute_shader_0 = compileProgram(
            compileShader(open("runtime/compute_0_init_boundary_particles.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 1
        self.compute_shader_1 = compileProgram(
            compileShader(open("runtime/compute_1_init_domain_particles.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 1i
        self.compute_shader_1i = compileProgram(
            compileShader(open("runtime/compute_1_init_inlet_particles.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 2a
        self.compute_shader_2a = compileProgram(
            compileShader(open("runtime/compute_2_boundary_density_pressure_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 2
        self.compute_shader_2 = compileProgram(
            compileShader(open("runtime/compute_2_density_pressure_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 2b
        self.compute_shader_2b = compileProgram(
            compileShader(open("runtime/compute_2_density_derivative_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 3
        self.compute_shader_3 = compileProgram(
            compileShader(open("runtime/compute_3_force_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 4
        self.compute_shader_4 = compileProgram(
            compileShader(open("runtime/compute_4_integrate_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 5
        self.compute_shader_5 = compileProgram(
            compileShader(open("runtime/compute_5_voxel_upgrade_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 6
        self.compute_shader_6 = compileProgram(
            compileShader(open("runtime/compute_6_combine_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # # compute shader a
        # self.compute_shader_a = compileProgram(
        #     compileShader(open("./MovingBoundaryShaders/compute_a_moving_boundary.shader", "rb"), GL_COMPUTE_SHADER))
        # glUseProgram(self.compute_shader_a)
        # self.compute_shader_a_current_step_loc = glGetUniformLocation(self.compute_shader_a, "current_step")
        # glUniform1i(self.compute_shader_a_current_step_loc, 0)

        # render shader
        self.render_shader = compileProgram(compileShader(open("runtime/vertex.shader", "rb"), GL_VERTEX_SHADER),
                                            compileShader(open("runtime/fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader)
        self.projection_loc = glGetUniformLocation(self.render_shader, "projection")
        self.view_loc = glGetUniformLocation(self.render_shader, "view")
        self.render_shader_color_type_loc = glGetUniformLocation(self.render_shader, "color_type")

        glUniform1i(self.render_shader_color_type_loc, 0)
        # render shader vector
        self.render_shader_vector = compileProgram(compileShader(open("runtime/vector_vertex.shader", "rb"), GL_VERTEX_SHADER),
                                                   compileShader(open("runtime/vector_geometry.shader", "rb"), GL_GEOMETRY_SHADER),
                                                   compileShader(open("runtime/vector_fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_vector)
        self.vector_projection_loc = glGetUniformLocation(self.render_shader_vector, "projection")
        self.vector_view_loc = glGetUniformLocation(self.render_shader_vector, "view")
        self.render_shader_vector_vector_type_loc = glGetUniformLocation(self.render_shader_vector, "vector_type")

        glUniform1i(self.render_shader_vector_vector_type_loc, 0)



        # render shader boundary
        self.render_shader_boundary = compileProgram(compileShader(open("runtime/boundary_vertex.shader", "rb"), GL_VERTEX_SHADER),
                                                     compileShader(open("runtime/fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_boundary)
        self.boundary_projection_loc = glGetUniformLocation(self.render_shader_boundary, "projection")
        self.boundary_view_loc = glGetUniformLocation(self.render_shader_boundary, "view")
        #

        # # compute shader for voxel debug
        # self.compute_shader_voxel = compileProgram(
        #     compileShader(open("voxel_compute.shader", "rb"), GL_COMPUTE_SHADER))
        # glUseProgram(self.compute_shader_voxel)
        # self.compute_shader_voxel_id_loc = glGetUniformLocation(self.compute_shader_voxel, "id")
#
        # glUniform1i(self.compute_shader_voxel_id_loc, 0)
        # render shader for voxel
        self.render_shader_voxel = compileProgram(compileShader(open("runtime/voxel_vertex.shader", "rb"), GL_VERTEX_SHADER),
                                                  compileShader(open("runtime/voxel_geometry.shader", "rb"), GL_GEOMETRY_SHADER),
                                                  compileShader(open("runtime/voxel_fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_voxel)

        self.voxel_projection_loc = glGetUniformLocation(self.render_shader_voxel, "projection")
        self.voxel_view_loc = glGetUniformLocation(self.render_shader_voxel, "view")

    def __call__(self, i, pause=False, show_vector=False, show_voxel=False, show_boundary=False):
        if self.need_init:
            self.need_init = False
            glUseProgram(self.compute_shader_0)
            glDispatchCompute(self.boundary_particle_number//4+1, 2, 2)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_1)
            glDispatchCompute(self.particle_number, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_1i)
            glDispatchCompute(self.inlet_particle_number, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        if not pause:
            glUseProgram(self.compute_shader_2a)
            glDispatchCompute(self.boundary_particle_number//4+1, 2, 2)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_2)
            glDispatchCompute((self.particle_number+1000)//10000+1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # glUseProgram(self.compute_shader_2b)
            # glDispatchCompute(self.MAX_PARTICLE_NUMBER // 10000, 100, 100)
            # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_3)
            glDispatchCompute((self.particle_number+1000)//10000+1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_4)
            glDispatchCompute((self.particle_number+1000)//10000+1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_5)
            glDispatchCompute(self.voxel_number//9+1, 3, 3)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_6)
            glDispatchCompute(self.voxel_number // 9 + 1, 3, 3)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # pull global_status buffer back and add delta_t*S*v
            status_float = np.empty((48,), dtype=np.byte)
            glGetNamedBufferSubData(self.sbo_global_status2, 0, 48, status_float)
            status_float = np.frombuffer(status_float, dtype=np.float32)
            status_float[9] += self.INLET1_FLUX/self.PARTICLE_VOLUME*self.DELTA_T
            status_float[10] += self.INLET2_FLUX/self.PARTICLE_VOLUME*self.DELTA_T
            status_float[11] += self.INLET3_FLUX/self.PARTICLE_VOLUME*self.DELTA_T

            status_int = np.empty((48,), dtype=np.byte)
            glGetNamedBufferSubData(self.sbo_global_status, 0, 48, status_int)
            status_int = np.frombuffer(status_int, dtype=np.int32)
            # mod inlet pointer by inlet number
            status_int[6] %= status_int[3]
            status_int[7] %= status_int[4]
            status_int[8] %= status_int[5]
            status_int[9] += int(status_float[9])
            status_int[10] += int(status_float[10])
            status_int[11] += int(status_float[11])

            status_float[9] -= int(status_float[9])
            status_float[10] -= int(status_float[10])
            status_float[11] -= int(status_float[11])
            glNamedBufferSubData(self.sbo_global_status2, 0, 48, status_float)
            glNamedBufferSubData(self.sbo_global_status, 0, 48, status_int)
            print(status_int[0], status_int[1])
            self.particle_number = status_int[0]
            # print(status_int[0])
            # print(status_int[9:12])
            # print(status_float[9:12])
            # print(status_int)
            # print("debug")
            # p = np.empty((640,), dtype=np.byte)
            # glGetNamedBufferSubData(self.sbo_particles, 0, 640, p)
            # p = np.frombuffer(p, dtype=np.float32).reshape((-1, 4))
            # v = np.empty((16*182*4,), dtype=np.byte)
            # glGetNamedBufferSubData(self.sbo_voxels_0, 124032*16*182*4, 16*182*4, v)
            # v = np.frombuffer(v, dtype=np.int32).reshape((-1, 4))
            # vr = np.empty((16 * 182 * 4,), dtype=np.byte)
            # glGetNamedBufferSubData(self.sbo_voxels_0, 124035 * 16 * 182 * 4, 16 * 182 * 4, vr)
            # vr = np.frombuffer(vr, dtype=np.int32).reshape((-1, 4))

            #         self.sbo_global_status = glGenBuffers(1)

        # glUseProgram(self.compute_shader_voxel)
        # glUniform1i(self.compute_shader_voxel_id_loc, i)
        # glDispatchCompute(1, 1, 1)
        # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindVertexArray(self.vao)

        if show_voxel:
            glUseProgram(self.render_shader_voxel)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2)
            glDrawArrays(GL_POINTS, 0, self.voxel_number)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if show_boundary:
            glUseProgram(self.render_shader_boundary)
            glPointSize(4)
            glDrawArrays(GL_POINTS, 0, self.boundary_particle_number)
            #

        glUseProgram(self.render_shader)
        glPointSize(2)
        glDrawArrays(GL_POINTS, 0, self.particle_number+1000)

        if show_vector:
            glUseProgram(self.render_shader_vector)
            glLineWidth(1)
            glDrawArrays(GL_POINTS, 0, self.particle_number+1000)




