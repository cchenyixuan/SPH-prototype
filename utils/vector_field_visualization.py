import time

import glfw
from OpenGL.GL import *
from camera import Camera

import numpy as np


class Equation:
    def __init__(self):
        self.position = np.zeros((3,), dtype=np.float32)
        self.value = 0.0
        self.vector = np.zeros((3,), dtype=np.float32)


class VectorField:
    def __init__(self):
        ...


class Displayer:
    """
    Basic displayer object creates a window and handles rendering loop
    """

    def __init__(self, window_size=(1920, 1080)):
        glfw.init()
        self.window = glfw.create_window(*window_size, "Displayer", None, None)
        glfw.set_window_pos(self.window, 200, 200)
        glfw.hide_window(self.window)

    def __call__(self, render_function, init_function=None):
        glfw.make_context_current(self.window)
        glfw.show_window(self.window)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glfw.set_key_callback(self.window, self.keyboard_clb)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # render parts
            render_function()

            glfw.swap_buffers(self.window)
        glfw.terminate()

    @staticmethod
    def keyboard_clb(window, key, scancode, action, mods):
        if key == glfw.KEY_Q and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)


class VectorFieldDisplayer:
    """
    This is a tool drawing vectors with strength(color) and direction(arrow).
    Current version supports 2D vector-field.
    """

    def __init__(self):
        # displayer parameters
        self.viewport_size = [1920, 1080]  # width, height
        self.resolution = 0.01  # one pixel represent 'resolution' X 'resolution' rectangle
        self.interval = 20  # points interval in pixels
        self.focus = [0.0, 0.0]  # screen center coordinates
        self.arrow = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        self.camera = Camera()
        """
        TODO: setup canvas and fill in values of the vector field by calculate function in every grid-point(invocation)
        perform a test case.
         ________________________
        |           ^ y          |
        |           |            |
        |      -----|-----> x    |
        |           |            |
        |________________________|
        """
        #
        pass

    def initialize_function(self) -> None:
        """
        This function initializes shaders and cameras.
        :return: None
        """
        ...

    def render_function(self) -> None:
        """
        This function calls OpenGL to render scenes.
        :return: None
        """
        ...

    @staticmethod
    def color_gradient(value: float, min_val=0.0, max_val=1.0) -> tuple:
        value = (value - min_val) / (max_val - min_val)  # normalize
        red = 1.0
        green = min(2 - 2 * value, 1.0)
        blue = max(1 - 2 * value, 0.0)
        return red, green, blue  # white -> yellow -> orange -> red


if __name__ == "__main__":
    dp = Displayer()


    def render_function():
        glClearColor(*VectorFieldDisplayer.color_gradient(glfw.get_time() % 10.0, 0.0, 10.0), 1.0)


    dp(render_function)
