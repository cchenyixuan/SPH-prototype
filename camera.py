import pyrr
import numpy as np
import threading
import time


class Camera:
    def __init__(self):
        self.position = pyrr.Vector4([0.0, 0.0, 10.0, 1.0])
        self.front = pyrr.Vector4([0.0, 0.0, -1.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        self.lookat = pyrr.Vector4([0.0, 0.0, 0.0, 1.0])

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1920/1080, 0.001, 1000)
        self.view = pyrr.matrix44.create_look_at(self.position.xyz, self.front.xyz, self.up.xyz)
        self.translate = pyrr.matrix44.create_identity()
        self.rotate = pyrr.matrix44.create_identity()
        self.scale = pyrr.matrix44.create_identity()

        self.mouse_left = False
        self.mouse_middle = False
        self.mouse_right = False
        self.mouse_pos = pyrr.Vector3([0.0, 0.0, 0.0])

    def __call__(self, delta=pyrr.Vector3([0.0, 0.0, 0.0]), flag=None):
        # left: rotate/spin
        if flag == "left" and self.mouse_left:
            self.operate_rotate(delta)
        # middle: move
        if flag == "middle" and self.mouse_middle:
            self.operate_translate(delta)
        # wheel scroll
        if flag == "wheel":
            self.operate_zoom(delta)

        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position+self.front).xyz, self.up.xyz)

    def operate_rotate(self, delta):
        # spin
        if abs(self.mouse_pos.x) >= 800 or abs(self.mouse_pos.y) >= 800:
            rotation_matrix = pyrr.matrix44.create_from_axis_rotation(-np.sign(
                (pyrr.vector3.cross(pyrr.Vector3([self.mouse_pos.x + delta.x, self.mouse_pos.y + delta.y, 0.0]),
                                    pyrr.Vector3([self.mouse_pos.x, self.mouse_pos.y, 0.0])))[2]) * self.front.xyz,
                                                                      delta.z)
            self.up = rotation_matrix @ self.up
        # rotate
        else:
            rotation_matrix = pyrr.matrix44.create_from_axis_rotation(self.up.xyz, delta.x / 100) @ \
                              pyrr.matrix44.create_from_axis_rotation(
                                  pyrr.vector3.cross(self.front.xyz, self.up.xyz),
                                  delta.y / 100)
            self.position = rotation_matrix @ (self.position - pyrr.Vector4([*self.lookat.xyz, 0.0])) + pyrr.Vector4(
                [*self.lookat.xyz, 0.0])
            self.front = rotation_matrix @ self.front
            self.up = rotation_matrix @ self.up

    def operate_translate(self, delta):
        right = pyrr.vector3.normalize(pyrr.vector3.cross(self.front.xyz, self.up.xyz))
        pos_before = np.array([*self.position], dtype=np.float32)
        self.position += pyrr.Vector4([*right, 0.0]) * delta.x * 0.1 * min(1.0,
                                                                           np.linalg.norm(self.position - self.lookat))
        self.position += pyrr.Vector4([*self.up.xyz, 0.0]) * delta.y * 0.1 * min(1.0, np.linalg.norm(
            self.position - self.lookat))
        delta_p = self.position - pos_before
        self.lookat += delta_p
        self.front = pyrr.Vector4(
            [*pyrr.vector3.normalize(pyrr.Vector3([*self.lookat.xyz]) - pyrr.Vector3([*self.position.xyz])), 1.0])

    def operate_zoom(self, delta):
        def smooth_zoom():
            for i in range(20):
                self.position += self.front * delta.y * 0.02 * min(1.0, np.linalg.norm(
                    self.position - self.lookat))
                self.position = pyrr.Vector4([*self.position.xyz, 1.0])

                time.sleep(0.005)
                if np.linalg.norm(self.position - self.lookat) <= 0.0:
                    if delta.y >= 0:
                        return

        zoom_thread = threading.Thread(target=smooth_zoom)
        zoom_thread.start()

    def x_view(self):
        self.position = pyrr.Vector4([-np.linalg.norm(self.position.xyz), 0.0, 0.0, 1.0])
        self.front = pyrr.Vector4([1.0, 0.0, 0.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def y_view(self):
        self.position = pyrr.Vector4([0.0, -np.linalg.norm(self.position.xyz), 0.0, 1.0])
        self.front = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        self.up = pyrr.Vector4([0.0, 0.0, 1.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def z_view(self):
        self.position = pyrr.Vector4([0.0, 0.0, -np.linalg.norm(self.position.xyz), 1.0])
        self.front = pyrr.Vector4([0.0, 0.0, 1.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def set_view(self, import_str: str):
        try:
            self.indice = [item.split(",") for item in import_str.split("\n")]
            self.position = pyrr.Vector4([*[float(item) for item in self.indice[0]], 1.0])
            self.front = pyrr.Vector4([*[float(item) for item in self.indice[1]], 1.0])
            self.up = pyrr.Vector4([*[float(item) for item in self.indice[2]], 1.0])
        except AttributeError:
            self.position = pyrr.Vector4([0.0, 0.0, 30.0, 1.0])
            self.front = pyrr.Vector4([0.0, 0.0, -1.0, 1.0])
            self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz), \
               "{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}".format(*self.position.xyz, *self.front.xyz, *self.up.xyz)
