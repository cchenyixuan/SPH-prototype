import time
import threading

import pyrr
import numpy as np
from OpenGL.GL import *
import glfw
import wcsph.Demo as Demo
from camera import Camera
from PIL import Image
from Coordinates import Coord
import os
from utils.terminal import Console

console = False
console_buffer = """>>>"""


class DisplayPort:
    def __init__(self):
        # self.console = threading.Thread(target=Console(self))
        glfw.init()
        self.window = glfw.create_window(1920, 1080, "Console", None, None)
        glfw.set_window_pos(self.window, 0, 30)
        glfw.hide_window(self.window)
        print("DisplayPort Initialized.")
        self.cursor_position = (0.0, 0.0)
        self.offset = 0
        self.left_click = False
        self.right_click = False
        self.middle_click = False
        self.pause = True
        self.show_vector = False
        self.show_boundary = False
        self.show_voxel = False
        self.record = False
        self.axis = True
        self.current_step = 0
        self.counter = 0
        self.camera = Camera()

        self.view = self.camera()[-1]
        self.view_changed = False

        os.makedirs(r"./tmp", exist_ok=True)

    def __call__(self, *args, **kwargs):
        glfw.make_context_current(self.window)
        # self.console.start()
        self.coordinates = Coord()
        self.three_d_cursor = Coord(r"Components/3d_cursor.obj")

        self.demo = Demo.Demo()
        glUseProgram(self.demo.render_shader_voxel)
        glUniformMatrix4fv(self.demo.voxel_projection_loc, 1, GL_FALSE, self.camera.projection)
        glUniformMatrix4fv(self.demo.voxel_view_loc, 1, GL_FALSE, self.camera.view)
        glUseProgram(self.demo.render_shader)
        glUniformMatrix4fv(self.demo.projection_loc, 1, GL_FALSE, self.camera.projection)
        glUniformMatrix4fv(self.demo.view_loc, 1, GL_FALSE, self.camera.view)
        glUseProgram(self.demo.render_shader_boundary)
        glUniformMatrix4fv(self.demo.boundary_projection_loc, 1, GL_FALSE, self.camera.projection)
        glUniformMatrix4fv(self.demo.boundary_view_loc, 1, GL_FALSE, self.camera.view)
        glUseProgram(self.demo.render_shader_vector)
        glUniformMatrix4fv(self.demo.vector_projection_loc, 1, GL_FALSE, self.camera.projection)
        glUniformMatrix4fv(self.demo.vector_view_loc, 1, GL_FALSE, self.camera.view)
        glUseProgram(self.demo.render_shader_boundary_vector)
        glUniformMatrix4fv(self.demo.boundary_vector_projection_loc, 1, GL_FALSE, self.camera.projection)
        glUniformMatrix4fv(self.demo.boundary_vector_view_loc, 1, GL_FALSE, self.camera.view)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)

        #glfw.window_hint(glfw.SAMPLES, 4)
        #glEnable(GL_MULTISAMPLE)

        self.track_cursor()
        self.track_keyboard()

        glfw.show_window(self.window)
        i = 0
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # coord
            if self.axis:
                self.coordinates(projection_matrix=self.camera.projection, view_matrix=self.view,
                                 model_matrix=pyrr.matrix44.create_identity(np.float32))
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                self.three_d_cursor(projection_matrix=self.camera.projection, view_matrix=self.view,
                                    model_matrix=pyrr.matrix44.create_from_translation(self.camera.lookat))
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # render codes
            self.demo(self.current_step, pause=self.pause, show_vector=self.show_vector, show_boundary=self.show_boundary,
                      show_voxel=self.show_voxel)
            # export boundary_data
            # if self.current_step % self.demo.save_frequency == 0 and self.current_step != 0:
            #     print("current step: ", self.current_step)
            #     self.save_data()
            if not self.pause:
                self.current_step += 1
            # camera update
            if self.view_changed:
                glProgramUniformMatrix4fv(self.demo.render_shader_voxel, self.demo.voxel_view_loc, 1, GL_FALSE,
                                          self.view)
                glProgramUniformMatrix4fv(self.demo.render_shader, self.demo.view_loc, 1, GL_FALSE, self.view)
                glProgramUniformMatrix4fv(self.demo.render_shader_boundary, self.demo.boundary_view_loc, 1, GL_FALSE, self.view)
                glProgramUniformMatrix4fv(self.demo.render_shader_vector, self.demo.vector_view_loc, 1, GL_FALSE,
                                          self.view)
                glProgramUniformMatrix4fv(self.demo.render_shader_boundary_vector, self.demo.boundary_vector_view_loc, 1, GL_FALSE,
                                          self.view)
                self.view_changed = False
            # animation
            if self.record:
                if self.current_step % 20 == 0:
                    try:
                        self.save_frames(f"tmp/{self.current_step // 20}.jpg")
                    except PermissionError:
                        pass
                # self.save_particle_data(i)
                i += 1

            glClearColor(0.0, 0.0, 0.0, 1.0)
            glfw.swap_buffers(self.window)
            # self.pause = True
        glfw.terminate()

    def save_data(self):
        # data = np.empty((self.demo.boundary_particles.nbytes,), dtype=np.byte)
        #
        # glGetNamedBufferSubData(self.demo.sbo_boundary_particles, 0, self.demo.boundary_particles.nbytes, data)
        # data = np.frombuffer(data, dtype=np.float32)
        # np.save(f"{self.current_step*self.demo.DELTA_T}.npy", data)
        ...

    def save_particle_data(self, i):
        import os
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/group1", exist_ok=True)
        os.makedirs("output/group2", exist_ok=True)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.demo.sbo_particles)
        buffer = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.demo.particles.nbytes),
                               dtype=np.float32).reshape((-1, 4))
        ptr = buffer.shape[0] // 2
        self.save_as_ply(buffer[:ptr], f"output/group1/{i}.ply")
        self.save_as_ply(buffer[ptr:], f"output/group2/{i}.ply")

    @staticmethod
    def save_as_ply(particles, export_path):
        header = f"""ply\nformat ascii 1.0\nelement vertex {particles.shape[0] // 4}\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar uint vertex_indices\nend_header\n"""
        with open(export_path, "w") as f:
            f.writelines(header)
            for i in range(particles.shape[0] // 4):
                f.write("{} {} {}\n".format(*particles[i * 4, :3]))
            f.close()

    @staticmethod
    def save_frames(filepath):
        x, y, width, height = glGetDoublev(GL_VIEWPORT)
        width, height = int(width), int(height)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filepath, "JPEG")

    def track_cursor(self):
        def cursor_position_clb(*args):
            delta = np.array(args[1:], dtype=np.float32) - self.cursor_position[:]
            self.cursor_position = args[1:]
            if self.left_click:
                self.view = self.camera(pyrr.Vector3((*delta, 0.0)) * 0.1, "left")[-1]
                self.view_changed = True
            elif self.middle_click:
                self.view = self.camera(pyrr.Vector3((-delta[0] * 0.01, delta[1] * 0.01, 0.0)), "middle")[-1]
                self.view_changed = True

        def mouse_press_clb(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
                self.left_click = True
                self.camera.mouse_left = True
            elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
                self.left_click = False
                self.camera.mouse_left = False
            if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
                self.right_click = True
                self.camera.mouse_right = True
            elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
                self.right_click = False
                self.camera.mouse_right = False
            if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
                self.middle_click = True
                self.camera.mouse_middle = True
            elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
                self.middle_click = False
                self.camera.mouse_middle = False

        def scroll_clb(window, x_offset, y_offset):
            self.view = self.camera(pyrr.Vector3((x_offset, y_offset, 0.0)), "wheel")[-1]
            self.view_changed = True

            def zoom():
                for i in range(20):
                    self.view = self.camera()[-1]
                    self.view_changed = True
                    time.sleep(0.005)

            t = threading.Thread(target=zoom)
            t.start()

        glfw.set_mouse_button_callback(self.window, mouse_press_clb)
        glfw.set_scroll_callback(self.window, scroll_clb)
        glfw.set_cursor_pos_callback(self.window, cursor_position_clb)

    def track_keyboard(self):
        def key_press_clb(window, key, scancode, action, mods):
            if key == glfw.KEY_SPACE and action == glfw.PRESS:
                self.pause = not self.pause
                if self.pause:
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.demo.sbo_particles)
                    a0 = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 5000000 * 64),
                                       dtype=np.float32)
                    self.a = np.reshape(a0, (-1, 4, 4))
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.demo.sbo_particles_sub_data)
                    a1 = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 5000000 * 64),
                                       dtype=np.float32)
                    self.b = np.reshape(a1, (-1, 4, 4))
                    print(np.hstack((self.a[:4], self.b[:4])))
                    # print(f"Total Particle: {sum([True if item[0, 3] else False for item in self.a.reshape((-1, 4, 4))])}")
                    # print(f"Largest Index: {np.max([step for step, item in enumerate(self.a.reshape((-1, 4, 4))) if item[0, 3] != 0])}")
                    # maxi = 0.0
                    # maxi_sample = 0.0
                    # for item in self.a.reshape((-1, 4, 4)):
                    #     if abs(item[1, 2])+abs(item[1, 3]) > maxi:
                    #         maxi = abs(item[1, 2])+abs(item[1, 3])
                    #         maxi_sample = item
                    # print(maxi)
                    # print(maxi_sample)
                    # print("over")
            if key == glfw.KEY_ENTER and action == glfw.PRESS:
                self.counter += 1
                self.counter %= self.demo.voxel_number
            if key == glfw.KEY_V and action == glfw.PRESS:
                self.show_vector = not self.show_vector
            if key == glfw.KEY_A and action == glfw.PRESS:
                if self.show_vector:
                    glProgramUniform1i(self.demo.render_shader_vector, self.demo.render_shader_vector_vector_type_loc,
                                       1)
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 1)
            if key == glfw.KEY_S and action == glfw.PRESS:
                if self.show_vector:
                    glProgramUniform1i(self.demo.render_shader_vector, self.demo.render_shader_vector_vector_type_loc,
                                       0)
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 0)
            if key == glfw.KEY_N and action == glfw.PRESS:
                if self.show_vector:
                    glProgramUniform1i(self.demo.render_shader_vector, self.demo.render_shader_vector_vector_type_loc,
                                       9)
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 9)
            if key == glfw.KEY_P and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 2)
            if key == glfw.KEY_D and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 3)
            if key == glfw.KEY_K and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 4)
            if key == glfw.KEY_U and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 5)
            if key == glfw.KEY_W and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 6)
            if key == glfw.KEY_T and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 7)

            if key == glfw.KEY_0 or key == glfw.KEY_KP_0 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 0)
            if key == glfw.KEY_1 or key == glfw.KEY_KP_1 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 1)
            if key == glfw.KEY_2 or key == glfw.KEY_KP_2 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 2)
            if key == glfw.KEY_3 or key == glfw.KEY_KP_3 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 3)
            if key == glfw.KEY_4 or key == glfw.KEY_KP_4 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 4)
            if key == glfw.KEY_5 or key == glfw.KEY_KP_5 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 5)
            if key == glfw.KEY_6 or key == glfw.KEY_KP_6 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 6)
            if key == glfw.KEY_7 or key == glfw.KEY_KP_7 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 7)
            if key == glfw.KEY_8 or key == glfw.KEY_KP_8 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 8)
            if key == glfw.KEY_9 or key == glfw.KEY_KP_9 and action == glfw.PRESS:
                glProgramUniform1i(self.demo.render_shader, self.demo.render_shader_color_type_loc, 9)

            if key == glfw.KEY_B and action == glfw.PRESS:
                self.show_boundary = not self.show_boundary
            if key == glfw.KEY_G and action == glfw.PRESS:
                self.show_voxel = not self.show_voxel
            if key == glfw.KEY_R and action == glfw.PRESS:
                self.record = not self.record
                if self.record:
                    print("Recording in progress.")
                else:
                    print("Recording stopped.")
            if key == glfw.KEY_C and action == glfw.PRESS:
                self.axis = not self.axis

        glfw.set_key_callback(self.window, key_press_clb)

    @staticmethod
    def get_gpu_data(buffer, offset, size, dtype):
        data = np.empty((size,), dtype=np.byte)
        glGetNamedBufferSubData(buffer, offset, size, data)
        data = np.frombuffer(data, dtype=dtype)
        return data


if __name__ == "__main__":
    dp = DisplayPort()
    dp()
