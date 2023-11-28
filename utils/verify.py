import numpy as np

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import time
from utils.shader_auto_completion import complete_shader

from project_loader import Project


class TestFunction:
    def __init__(self):
        self.x = np.linspace(-3, 3, 901)
        self.y = np.linspace(-3, 3, 901)
        self.buffer = np.zeros((901*901, 4, 4), dtype=np.float32)
        for i in range(self.buffer.shape[0]):
            self.buffer[i][0, 0], self.buffer[i][0, 2] = self.x[i // 901], self.y[i % 901]
            self.buffer[i][3, 0] = self.test_function(self.buffer[i][0, 0], self.buffer[i][0, 2])
            self.buffer[i][2, 2] = self.test_function(self.buffer[i][0, 0], self.buffer[i][0, 2])
            self.buffer[i][3, 2:] = self.gradient_test_function(self.buffer[i][0, 0], self.buffer[i][0, 2])
            self.buffer[i][3, 1] = self.laplacian_test_function(self.buffer[i][0, 0], self.buffer[i][0, 2])

    def __call__(self, *args, **kwargs):
        return self.buffer

    @staticmethod
    def test_function(x, y):
        t = x ** 2 + y ** 2
        return np.cos(t) * np.exp(-t)

    @staticmethod
    def gradient_test_function(x, y):
        t = x ** 2 + y ** 2
        return np.array([-np.sin(t) * np.exp(-t) * 2 * x - 2 * x * np.cos(t) * np.exp(-t),
                         -np.sin(t) * np.exp(-t) * 2 * y - 2 * y * np.cos(t) * np.exp(-t)], dtype=np.float32)

    @staticmethod
    def laplacian_test_function(x, y):
        t = x ** 2 + y ** 2
        return np.exp(-t) * (8 * t * np.sin(t) - 4 * np.sin(t) - 4 * np.cos(t))


class Demo:
    def __init__(self):
        # --case parameters--
        self.H = 0.1
        self.R = 0.01/3
        self.DELTA_T = 0.0000025
        self.PARTICLE_VOLUME = 0.01 * 0.01*1.6/1.4243242  # ~69 points inside a circle with radius H

        self.voxel_buffer_file = r"D:\ProgramFiles\PycharmProject\VoxelizationAlg\voxelization\buffer.npy"
        self.voxel_origin_offset = [-3., -0.0099, -3.]
        self.domain_particle_file = TestFunction()()

        # --solver parameters--
        self.VOXEL_MEMORY_LENGTH = 2912  # (2+60+60+60)*16
        self.VOXEL_BLOCK_SIZE = 960  # 60*16
        self.VOXEL_GROUP_SIZE = 300000  # one ssbo could get unstable if size is larger than (2912*4bit)*300000
        self.VOXEL_MATRIX_NUMBER = 182  # 2+60+60+60
        self.MAX_PARTICLE_NUMBER = 5000000

        def create_particle_sub_buffer(particles, group_id):
            buffer = np.zeros_like(particles)
            for i in range(buffer.shape[0] // 4):
                buffer[i * 4 + 3][-1] = group_id
            return buffer

        self.particles = self.domain_particle_file.reshape((-1, 4))

        self.particles_sub_data = create_particle_sub_buffer(self.particles, 0)
        self.particle_number = self.particles.shape[0] // 4  # (n * 4, 4)

        self.voxels = np.load(self.voxel_buffer_file)  # np.load(self.voxel_buffer_file)
        self.voxel_number = self.voxels.shape[0] // (self.VOXEL_MATRIX_NUMBER * 4)  # (n * (182*4), 4)
        # for i in range(self.voxel_number):
        #     self.voxels[182*4*i+8: 182*4*i+28] = 0.0
        self.voxel_groups = [self.voxels[
                             self.VOXEL_MATRIX_NUMBER * 4 * self.VOXEL_GROUP_SIZE * i:self.VOXEL_MATRIX_NUMBER * 4 * self.VOXEL_GROUP_SIZE * (
                                         i + 1)] for i in range(self.voxel_number // self.VOXEL_GROUP_SIZE)]
        if self.voxel_number % self.VOXEL_GROUP_SIZE != 0:
            self.voxel_groups.append(self.voxels[self.VOXEL_MATRIX_NUMBER * 4 * self.VOXEL_GROUP_SIZE * (
                        self.voxel_number // self.VOXEL_GROUP_SIZE):])
        print(f"{len(self.voxel_groups)} voxel groups created.")
        self.voxel_particle_numbers = np.zeros((self.voxel_number,), dtype=np.int32)

        # initialize OpenGL
        # particles buffer
        self.sbo_particles = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_particles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.sbo_particles)
        glNamedBufferStorage(self.sbo_particles, np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32).nbytes,
                             np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32), GL_DYNAMIC_STORAGE_BIT)
        glNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, self.particles)

        # particles sub data buffer
        self.sbo_particles_sub_data = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_particles_sub_data)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.sbo_particles_sub_data)
        glNamedBufferStorage(self.sbo_particles_sub_data,
                             np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32).nbytes,
                             np.zeros((self.MAX_PARTICLE_NUMBER, 16), dtype=np.float32), GL_DYNAMIC_STORAGE_BIT)
        glNamedBufferSubData(self.sbo_particles_sub_data, 0, self.particles_sub_data.nbytes, self.particles_sub_data)

        # voxel_particle_numbers buffer
        self.sbo_voxel_particle_numbers = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_voxel_particle_numbers)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.sbo_voxel_particle_numbers)
        glNamedBufferStorage(self.sbo_voxel_particle_numbers, self.voxel_particle_numbers.nbytes,
                             self.voxel_particle_numbers, GL_DYNAMIC_STORAGE_BIT)

        # voxels buffer
        for index, buffer in enumerate(self.voxel_groups):
            self.__setattr__(f"sbo_voxels_{index}", glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.__getattribute__(f"sbo_voxels_{index}"))
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10 + index, self.__getattribute__(f"sbo_voxels_{index}"))
            glNamedBufferStorage(self.__getattribute__(f"sbo_voxels_{index}"), buffer.nbytes, buffer,
                                 GL_DYNAMIC_STORAGE_BIT)

        self.indices_buffer = np.array([i for i in range(max(self.voxel_number, self.MAX_PARTICLE_NUMBER))],
                                       dtype=np.int32)

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

        # compute shader 1
        self.compute_shader_1 = compileProgram(
            compileShader(open(r".\KS\solvers-2d\compute_1_init_domain_particles.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 2
        self.compute_shader_2 = compileProgram(
            compileShader(open(r".\KS\solvers-2d\compute_2_grad_laplacian_solver.shader", "rb"), GL_COMPUTE_SHADER))
        # compute shader 3
        self.compute_shader_3 = compileProgram(
            compileShader(open(r".\KS\solvers-2d\compute_3_integrate_solver.shader", "rb"), GL_COMPUTE_SHADER))

        # render shader
        self.render_shader = compileProgram(
            compileShader(open(r"./KS/KS_shaders/vertex.shader", "rb"), GL_VERTEX_SHADER),
            compileShader(open(r"./KS/KS_shaders/fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader)
        self.projection_loc = glGetUniformLocation(self.render_shader, "projection")
        self.view_loc = glGetUniformLocation(self.render_shader, "view")
        self.render_shader_color_type_loc = glGetUniformLocation(self.render_shader, "color_type")

        glUniform1i(self.render_shader_color_type_loc, 0)
        # render shader vector
        self.render_shader_vector = compileProgram(
            compileShader(open(r"./KS/KS_shaders/vector_vertex.shader", "rb"), GL_VERTEX_SHADER),
            compileShader(open(r"./KS/KS_shaders/vector_geometry.shader", "rb"), GL_GEOMETRY_SHADER),
            compileShader(open(r"./KS/KS_shaders/vector_fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_vector)
        self.vector_projection_loc = glGetUniformLocation(self.render_shader_vector, "projection")
        self.vector_view_loc = glGetUniformLocation(self.render_shader_vector, "view")
        self.render_shader_vector_vector_type_loc = glGetUniformLocation(self.render_shader_vector, "vector_type")

        glUniform1i(self.render_shader_vector_vector_type_loc, 0)

        # # compute shader for voxel debug
        # self.compute_shader_voxel = compileProgram(
        #     compileShader(open("voxel_compute.shader", "rb"), GL_COMPUTE_SHADER))
        # glUseProgram(self.compute_shader_voxel)
        # self.compute_shader_voxel_id_loc = glGetUniformLocation(self.compute_shader_voxel, "id")
        #
        # glUniform1i(self.compute_shader_voxel_id_loc, 0)
        # render shader for voxel
        self.render_shader_voxel = compileProgram(
            compileShader(open(r"./KS/KS_shaders/voxel_vertex.shader", "rb"), GL_VERTEX_SHADER),
            compileShader(open(r"./KS/KS_shaders/voxel_geometry.shader", "rb"), GL_GEOMETRY_SHADER),
            compileShader(open(r"./KS/KS_shaders/voxel_fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_voxel)

        self.voxel_projection_loc = glGetUniformLocation(self.render_shader_voxel, "projection")
        self.voxel_view_loc = glGetUniformLocation(self.render_shader_voxel, "view")

    def __call__(self, i, pause=False, show_vector=False, show_voxel=False, show_boundary=False):
        if self.need_init:
            self.need_init = False

            glUseProgram(self.compute_shader_1)
            glDispatchCompute(self.particle_number//10000+1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_2)
            glDispatchCompute(self.particle_number // 10000 + 1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            tmp = np.empty_like(self.particles)
            glGetNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, tmp)
            tmp = tmp.reshape((-1, 4, 4))
            self.particles = tmp

            print("init over")
        if not pause:
            glUseProgram(self.compute_shader_2)
            glDispatchCompute(self.particle_number // 10000 + 1, 100, 100)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            tmp = np.empty_like(self.particles)
            glGetNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, tmp)
            tmp = tmp.reshape((-1, 4, 4))
            self.particles = tmp

        glBindVertexArray(self.vao)

        if show_voxel:
            glUseProgram(self.render_shader_voxel)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2)
            glDrawArrays(GL_POINTS, 0, self.voxel_number)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if show_boundary:
            ...

        glUseProgram(self.render_shader)
        glPointSize(2)
        glDrawArrays(GL_POINTS, 0, self.particle_number)

        if show_vector:
            glUseProgram(self.render_shader_vector)
            glLineWidth(1)
            glDrawArrays(GL_POINTS, 0, self.particle_number)
