import time

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw


def test_matrix_multiple(matrix_a: np.ndarray, matrix_b):
    # pad to be 4x
    matrix_a = np.vstack((matrix_a, np.zeros(((4 - matrix_a.shape[0] % 4) % 4, matrix_a.shape[1]))))
    matrix_a = np.hstack((matrix_a, np.zeros((matrix_a.shape[0], (4 - matrix_a.shape[1] % 4) % 4))))

    matrix_b = np.vstack((matrix_b, np.zeros(((4 - matrix_b.shape[0] % 4) % 4, matrix_b.shape[1]))))
    matrix_b = np.hstack((matrix_b, np.zeros((matrix_b.shape[0], (4 - matrix_b.shape[1] % 4) % 4))))
    # calculate shape
    shape_array = np.array([matrix_a.shape[0] // 4, matrix_a.shape[1] // 4, matrix_b.shape[1] // 4, 0], dtype=np.uint32)
    # re-arrange a and b
    matrix_a = np.vstack(
        [item for v_block in np.vsplit(matrix_a, shape_array[0]) for item in np.hsplit(v_block, shape_array[1])])
    matrix_b = np.vstack(
        [item for v_block in np.vsplit(matrix_b, shape_array[1]) for item in np.hsplit(v_block, shape_array[2])])
    matrix_c = np.zeros((shape_array[0] * 4, shape_array[2] * 4), dtype=np.float64)
    # prepare opengl content
    glfw.init()
    window = glfw.create_window(100, 100, "Console", None, None)
    glfw.hide_window(window)
    glfw.make_context_current(window)
    compute_shader = compileProgram(compileShader(open("matrix_dot.shader", "rb"), GL_COMPUTE_SHADER))
    glProgramUniform4uiv(compute_shader, glGetUniformLocation(compute_shader, "shapes"), 1, shape_array)
    # send buffer to GPU
    sbo_matrix_a = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo_matrix_a)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sbo_matrix_a)
    glNamedBufferStorage(sbo_matrix_a, matrix_a.nbytes, matrix_a, GL_DYNAMIC_STORAGE_BIT)
    glNamedBufferSubData(sbo_matrix_a, 0, matrix_a.nbytes, matrix_a)
    sbo_matrix_b = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo_matrix_b)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sbo_matrix_b)
    glNamedBufferStorage(sbo_matrix_b, matrix_b.nbytes, matrix_b, GL_DYNAMIC_STORAGE_BIT)
    glNamedBufferSubData(sbo_matrix_b, 0, matrix_b.nbytes, matrix_b)
    sbo_matrix_c = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo_matrix_c)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sbo_matrix_c)
    glNamedBufferStorage(sbo_matrix_c, matrix_c.nbytes, matrix_c, GL_DYNAMIC_STORAGE_BIT)
    glNamedBufferSubData(sbo_matrix_c, 0, matrix_c.nbytes, matrix_c)
    # compute
    s = time.time()
    a = 0
    while True:
        glfw.poll_events()
        # dispatch jobs
        glUseProgram(compute_shader)
        glDispatchCompute(shape_array[0], 1, shape_array[2])
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        a += 1
        if a == 1:
            # collect data
            print(time.time() - s)
            buffer = np.empty((matrix_c.nbytes,))
            print(time.time() - s)
            glGetNamedBufferSubData(sbo_matrix_c, 0, matrix_c.nbytes, buffer)
            print(time.time() - s)
            matrix_c = np.frombuffer(buffer, dtype=np.float64).reshape((-1, 4, 4))
            # re-arrange data
            matrix_c = np.block(
                [[matrix_c[i * shape_array[2] + j] for j in range(shape_array[2])] for i in range(shape_array[0])])
            break

    # trash collect
    glfw.terminate()
    return matrix_c


if __name__ == "__main__":
    m, n, k = 2000, 2000, 2000
    a = np.array([i for i in range(m * n)], dtype=np.float64).reshape((m, n))
    b = np.array([i for i in range(n * k)], dtype=np.float64).reshape((n, k))
    c = test_matrix_multiple(a, b)
    s = time.time()
    cc = a @ b
    print("np: ", time.time() - s)
    print(np.all(c[:m, :k] == cc))
