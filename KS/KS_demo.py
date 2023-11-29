import numpy as np

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import time
from utils.shader_auto_completion import complete_shader

from project_loader import Project
from multiprocessing import Pool


class PoissonSolver:
    def __init__(self, f: np.ndarray):
        self.m = f.shape[0]
        self.h = 1/(1+self.m)
        self.f = 0.0025**2*f
        self.s = np.sin(np.array([[i*j*np.pi*self.h for j in range(1, self.m+1)] for i in range(1, self.m+1)], dtype=np.float32))
        self.sigma = np.array([np.sin(i*np.pi*self.h/2)**2 for i in range(1, self.m+1)], dtype=np.float32)
        self.g = self.s@self.f@self.s
        self.x = np.array([[self.h**2*self.g[i-1, j-1]/(self.sigma[i-1]+self.sigma[j-1]) for j in range(1, self.m+1)] for i in range(1, self.m+1)], dtype=np.float32)
        self.v = self.s@self.x@self.s

    def __call__(self, f):
        self.f = 0.0025 ** 2 * f
        self.g = self.s @ self.f @ self.s
        self.x = np.array([[self.h ** 2 * self.g[i - 1, j - 1] / (
                    self.sigma[i - 1] + self.sigma[j - 1]) for j in range(1, self.m + 1)] for i in
                           range(1, self.m + 1)], dtype=np.float32)

        self.v = self.s @ self.x @ self.s
        return self.v

    def forward(self):
        self.forward_stencil = 2*np.eye(self.m)
        self.forward_stencil[0, 1] = -1
        self.forward_stencil[-1, -2] = -1
        for i in range(1, self.m-1):
            self.forward_stencil[i, i-1], self.forward_stencil[i, i+1] = -1, -1
        ans = (self.forward_stencil@self.v + self.v@self.forward_stencil)/0.0025**2
        return ans


class BesselSolver:
    def __init__(self, f: np.ndarray):
        self.m = f.shape[0]
        self.h = 1/(1+self.m)
        self.f = self.h**2*f
        self.s = np.sin(np.array([[i*j*np.pi*self.h for j in range(1, self.m+1)] for i in range(1, self.m+1)], dtype=np.float32))
        self.sigma = np.array([np.sin(i*np.pi*self.h/2)**2 for i in range(1, self.m+1)], dtype=np.float32)
        self.g = self.s@self.f@self.s
        self.x = np.array([[self.h**2*self.g[i-1, j-1]/(self.sigma[i-1]+self.sigma[j-1]) for j in range(1, self.m+1)] for i in range(1, self.m+1)], dtype=np.float32)
        self.v = self.s@self.x@self.s

    def __call__(self, f):
        self.f = self.h**2 * f
        self.g = self.s @ self.f @ self.s
        self.x = np.array([[self.h ** 2 * self.g[i - 1, j - 1] / (
                    self.sigma[i - 1] + self.sigma[j - 1]) for j in range(1, self.m + 1)] for i in
                           range(1, self.m + 1)], dtype=np.float32)

        self.v = self.s @ self.x @ self.s
        return self.v

    def forward(self):
        self.forward_stencil = 2*np.eye(self.m)
        self.forward_stencil[0, 1] = -1
        self.forward_stencil[-1, -2] = -1
        for i in range(1, self.m-1):
            self.forward_stencil[i, i-1], self.forward_stencil[i, i+1] = -1, -1
        ans = (self.forward_stencil@self.v + self.v@self.forward_stencil)/self.h**2
        return ans

def conv_worker(args):
    i, kernel, pad = args
    print(i - 250)
    ans_ = []
    for j in range(250, 250 + 2001):
        res = kernel * pad[i - 250:i + 251, j - 250:j + 251]
        ans_.append(np.sum(res) * 0.0001)
    return ans_
class BesselConv:
    def __init__(self):
        self.bessel_2d_data = np.load(r"D:\ProgramFiles\PycharmProject\SPH-prototype\KS/result2.npy")
        self.kernel = self.discrete_kernel()
        self.kernel = self.kernel[749:-749, 749:-749]

    def __call__(self, u):
        pad = np.zeros((u.shape[0] + 250 * 2, u.shape[1] + 250 * 2), dtype=np.float32)
        pad[250:250 + u.shape[0], 250:250 + u.shape[1]] = u

        pool = Pool(20)

        t = time.time()
        ans__ = pool.map(conv_worker, [(i, self.kernel, pad) for i in range(250, 250 + 2001)])
        print(time.time() - t)
        pool.close()
        return np.array(ans__, dtype=np.float32)

    def discrete_kernel(self):
        kernel = np.zeros((1999, 1999), dtype=np.float32)
        quarter_kernel = np.zeros((999, 999), dtype=np.float32)
        for i in range(1, 1000):
            for j in range(1, 1000):
                quarter_kernel[i-1, j-1] = self.get_kernel_value(i*0.01, j*0.01)
        line_kernel = np.zeros((999, ), dtype=np.float32)
        for i in range(1, 999):
            line_kernel[i-1] = self.get_kernel_value(i*0.01, 0.0)
        kernel[999, 1000:] = line_kernel
        kernel[999, :999] = line_kernel[::-1]
        kernel[:999, 999] = line_kernel[::-1]
        kernel[1000:, 999] = line_kernel

        kernel[1000:, 1000:] = quarter_kernel
        kernel[:999, 1000:] = quarter_kernel[::-1]
        kernel[:999, :999] = quarter_kernel[::-1, ::-1]
        kernel[1000:, :999] = quarter_kernel[:, ::-1]
        return kernel

    def get_kernel_value(self, x, y):
        norm = np.linalg.norm([x, y])
        if norm > 9.999:
            return 0.0
        a = int(norm // 0.001)
        b = a + 1
        theta = norm % 0.001 / 0.001
        return self.bessel_2d_data[1, a] * (1 - theta) + self.bessel_2d_data[1, b] * theta

class Demo:
    def __init__(self):
        # --case parameters--
        self.H = 0.05
        self.R = 0.005
        self.DELTA_T = 0.000005
        self.PARTICLE_VOLUME = 8.538886859432597e-05

        self.voxel_buffer_file = r".\v_buffer.npy"
        self.voxel_origin_offset = [-5.05, -0.05, -5.05]
        self.domain_particle_file = r".\p_buffer.npy"

        # --solver parameters--
        self.VOXEL_MEMORY_LENGTH = 2912  # (2+60+60+60)*16
        self.VOXEL_BLOCK_SIZE = 960  # 60*16
        self.VOXEL_GROUP_SIZE = 300000  # one ssbo could get unstable if size is larger than (2912*4bit)*300000
        self.VOXEL_MATRIX_NUMBER = 182  # 2+60+60+60
        self.MAX_PARTICLE_NUMBER = 5000000

        # prepare buffers
        def load_file(file):
            import re
            find_vertex = re.compile(r"v (\+?-?[\d.]+) (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
            data = []
            try:
                with open(file, "r") as f:
                    for row in f:
                        ans = re.findall(find_vertex, row)
                        if ans:
                            ans = [float(ans[0][i]) for i in range(3)]
                            data.append([ans[0], ans[1], ans[2]])
                    f.close()
            except FileNotFoundError:
                pass
            return np.array(data, dtype=np.float32)

        def load_domain(particles):
            output = np.zeros((particles.shape[0] * 4, 4), dtype=np.float32)
            tmp = 0
            for step, vertex in enumerate(particles):
                output[step * 4][:3] = vertex
                value = np.random.random(1)
                # max(0, 1.5-np.linalg.norm(vertex[:3]))
                output[step * 4+2][2:] = [max(0, 2*np.sin(4*np.pi*np.linalg.norm(vertex[:3]))), 0.0]
                tmp += self.PARTICLE_VOLUME*max(0, 2*np.sin(4*np.pi*np.linalg.norm(vertex[:3])))
            print("total u is :", tmp)
            return output

        def create_particle_sub_buffer(particles, group_id):
            buffer = np.zeros_like(particles)
            for i in range(buffer.shape[0] // 4):
                buffer[i * 4 + 3][-1] = group_id
                # div(u)
                buffer[i * 4 + 0][0] = 0.01/(1+particles[i * 4 + 0][0]**2+particles[i * 4 + 0][2]**2)**2
                # v0 = n

            return buffer

        def poly6_function_2d(rij, h):
            if rij > h:
                return 0.0
            return 4 / (np.pi * h**8) * (h*h - rij * rij)**3

        u_sum = 0.0
        self.particles = np.load(self.domain_particle_file).reshape((-1, 4))  # load_domain(load_file(self.domain_particle_file))
        # self.besselconv = BesselConv()
        # CALCULATE V BY BESSELSOLVER
        # u = np.zeros((2001, 2001), dtype=np.float32)
        # v = np.zeros((2001, 2001), dtype=np.float32)
        # for i in range(2001*2001):
        #     u[i//2001, i%2001] = self.particles[i*4+2, 2]
        #     v[i//2001, i%2001] = self.particles[i*4+2, 3]
        # print(np.sum(u), np.sum(v))
        # BS = BesselSolver(u)
        # vv = BS.v
        # print(np.sum(vv))
        # for i in range(2001*2001):
        #     # self.particles[i*4+3, 2] = uuu[i//2001, i%2001]
        #     self.particles[i * 4 + 2, 3] = vv[i // 2001, i % 2001]
        # self.particles = self.particles.reshape((-1, 4))
        self.particles_sub_data = create_particle_sub_buffer(self.particles, 0)
        gaussian = lambda x: 1/2*np.pi * np.exp(-x@x.T/2/0.1)
        for i in range(self.particles.shape[0] // 4):
            x = np.array([self.particles[i * 4 + 0][0], self.particles[i * 4 + 0][2]], dtype=np.float32)

            self.particles[i * 4 + 2][3] = gaussian(x)/gaussian(np.array((0, 0)))*200.0
        self.particle_number = self.particles.shape[0] // 4  # (n * 4, 4)

        self.voxels = np.load(self.voxel_buffer_file)#np.load(self.voxel_buffer_file)
        self.voxel_number = self.voxels.shape[0] // (self.VOXEL_MATRIX_NUMBER*4)  # (n * (182*4), 4)
        # for i in range(self.voxel_number):
        #     self.voxels[182*4*i+8: 182*4*i+28] = 0.0
        self.voxel_groups = [self.voxels[self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*i:self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*(i+1)] for i in range(self.voxel_number//self.VOXEL_GROUP_SIZE)]
        if self.voxel_number % self.VOXEL_GROUP_SIZE != 0:
            self.voxel_groups.append(self.voxels[self.VOXEL_MATRIX_NUMBER*4*self.VOXEL_GROUP_SIZE*(self.voxel_number//self.VOXEL_GROUP_SIZE):])
        print(f"{len(self.voxel_groups)} voxel groups created.")
        self.voxel_particle_numbers = np.zeros((self.voxel_number, ), dtype=np.int32)



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

        # voxel_particle_numbers buffer
        self.sbo_voxel_particle_numbers = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.sbo_voxel_particle_numbers)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.sbo_voxel_particle_numbers)
        glNamedBufferStorage(self.sbo_voxel_particle_numbers, self.voxel_particle_numbers.nbytes, self.voxel_particle_numbers, GL_DYNAMIC_STORAGE_BIT)

        # voxels buffer
        for index, buffer in enumerate(self.voxel_groups):
            self.__setattr__(f"sbo_voxels_{index}", glGenBuffers(1))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.__getattribute__(f"sbo_voxels_{index}"))
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10+index, self.__getattribute__(f"sbo_voxels_{index}"))
            glNamedBufferStorage(self.__getattribute__(f"sbo_voxels_{index}"), buffer.nbytes, buffer, GL_DYNAMIC_STORAGE_BIT)

        self.indices_buffer = np.array([i for i in range(max(self.voxel_number, self.MAX_PARTICLE_NUMBER))], dtype=np.int32)

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
        self.render_shader = compileProgram(compileShader(open(r"./KS/KS_shaders/vertex.shader", "rb"), GL_VERTEX_SHADER),
                                            compileShader(open(r"./KS/KS_shaders/geometry.shader", "rb"),
                                                          GL_GEOMETRY_SHADER),
                                            compileShader(open(r"./KS/KS_shaders/fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader)
        self.projection_loc = glGetUniformLocation(self.render_shader, "projection")
        self.view_loc = glGetUniformLocation(self.render_shader, "view")
        self.render_shader_color_type_loc = glGetUniformLocation(self.render_shader, "color_type")

        glUniform1i(self.render_shader_color_type_loc, 0)
        # render shader vector
        self.render_shader_vector = compileProgram(compileShader(open(r"./KS/KS_shaders/vector_vertex.shader", "rb"), GL_VERTEX_SHADER),
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
        self.render_shader_voxel = compileProgram(compileShader(open(r"./KS/KS_shaders/voxel_vertex.shader", "rb"), GL_VERTEX_SHADER),
                                                  compileShader(open(r"./KS/KS_shaders/voxel_geometry.shader", "rb"), GL_GEOMETRY_SHADER),
                                                  compileShader(open(r"./KS/KS_shaders/voxel_fragment.shader", "rb"), GL_FRAGMENT_SHADER))
        glUseProgram(self.render_shader_voxel)

        self.voxel_projection_loc = glGetUniformLocation(self.render_shader_voxel, "projection")
        self.voxel_view_loc = glGetUniformLocation(self.render_shader_voxel, "view")

    def __call__(self, i, pause=False, show_vector=False, show_voxel=False, show_boundary=False):
        if self.need_init:
            self.need_init = False
            # s = time.time()
            # glUseProgram(self.compute_shader_1)
            # glDispatchCompute(self.particle_number, 1, 1)
            # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            # buffer1 = np.empty_like(self.voxel_groups[0])
            # glGetNamedBufferSubData(self.sbo_voxels_0, 0, buffer1.nbytes, buffer1)
            # buffer1 = np.frombuffer(buffer1, dtype=np.int32).reshape((-1, 4))
            # # buffer2 = np.empty_like(self.voxel_groups[1])
            # # glGetNamedBufferSubData(self.sbo_voxels_1, 0, buffer2.nbytes, buffer2)
            # # buffer2 = np.frombuffer(buffer2, dtype=np.int32).reshape((-1, 4))
            # np.save("v_buffer.npy", buffer1)#np.vstack((buffer1, buffer2)))
            # print(f"{time.time() - s}s for init.")
            # print("init over")
            # point_buffer = np.empty_like(self.particles)
            # glGetNamedBufferSubData(self.sbo_particles, 0, point_buffer.nbytes, point_buffer)
            # point_buffer = np.frombuffer(point_buffer, dtype=np.float32).reshape((-1, 4))
            # np.save("p_buffer.npy", point_buffer)
            # print("init over")
        if not pause:
            glUseProgram(self.compute_shader_2)
            glDispatchCompute(self.particle_number, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(self.compute_shader_3)
            glDispatchCompute(self.particle_number, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # print out total u and v
            # tmp = np.empty_like(self.particles)
            # glGetNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, tmp)
            # self.umap = np.array([tmp[i * 4 + 2][2] for i in range(tmp.shape[0] // 4)],
            #                      dtype=np.float32).reshape([2001, 2001])
            # self.vmap = self.besselconv(self.umap)
            # for i in range(self.particles.shape[0] // 4):
            #     tmp[i * 4 + 2][3] = self.vmap[i // 2001, i % 2001]
            # self.particles = tmp
            # glNamedBufferSubData(self.sbo_particles, 0, self.particles.nbytes, self.particles)
#
            # tmp = np.frombuffer(tmp, dtype=np.float32).reshape((-1, 4, 4))
            # total_u, total_v = 0, 0
            # grad_u = 0
            # grad_v = 0
            # lap_u = 0
            # lap_v = 0
            # for item in tmp:
            #     total_u += item[2, 2]
            #     total_v += item[2, 3]
            #     grad_u = max(grad_u, abs(item[1, 0])+abs(item[1, 1]))
            #     grad_v = max(grad_v, abs(item[1, 2]) + abs(item[1, 3]))
            #     lap_u = max(lap_u, abs(item[2, 0]))
            #     lap_v = max(lap_v, abs(item[2, 1]))
            # print(f"Total u = {total_u*self.PARTICLE_VOLUME}, Total v = {total_v*self.PARTICLE_VOLUME}, max grad u = {grad_u}, max grad v = {grad_v}, max lap u = {lap_u}, max lap v = {lap_v}")



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




