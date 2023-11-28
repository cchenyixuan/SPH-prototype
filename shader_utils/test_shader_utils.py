import time

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw


def test_matrix_multiple(matrix_a: np.ndarray, matrix_b):
    # test A amd B
    shape_array = np.array([matrix_a.shape[0] // 4, matrix_a.shape[1] // 4, matrix_b.shape[1] // 4, 0], dtype=np.uint32)
    matrix_a = np.vstack([item for v_block in np.vsplit(matrix_a, shape_array[0]) for item in np.hsplit(v_block, shape_array[1])])
    matrix_b = np.vstack([item for v_block in np.vsplit(matrix_b, shape_array[1]) for item in np.hsplit(v_block, shape_array[2])])
    matrix_c = np.zeros((shape_array[0]*4, shape_array[2]*4), dtype=np.float32)
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
    aa = 0
    while True:
        aa += 1
        glfw.poll_events()
        s = time.time()
        glUseProgram(compute_shader)
        glDispatchCompute(*shape_array[:3])
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        buffer = np.empty((matrix_c.nbytes,), dtype=np.byte)

        glGetNamedBufferSubData(sbo_matrix_c, 0, matrix_c.nbytes, buffer)
        print(time.time() - s)
        matrix_c = np.frombuffer(buffer, dtype=np.float32).reshape((-1, 4, 4))
        matrix_c = np.block([[matrix_c[i*shape_array[2] + j] for j in range(shape_array[2])] for i in range(shape_array[0])])
        if aa > 10:
            break
    glfw.terminate()
    return matrix_c


if __name__ == "__main__":
    a = np.array([i for i in range(6400*8000)], dtype=np.float32).reshape((6400, 8000))

    b = np.array([i for i in range(8000*1200)], dtype=np.float32).reshape((8000, 1200))
    c = test_matrix_multiple(a, b)
    s = time.time()
    cc = a@b
    print("np: ", time.time()-s)
