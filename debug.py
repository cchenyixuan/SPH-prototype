import numpy as np
import math


class Voxelization:
    def __init__(self, vertex_buffer, voxel_length):
        self.vertex_buffer = vertex_buffer
        self.lower_bound = np.min(self.vertex_buffer, axis=0)
        self.upper_bound = np.max(self.vertex_buffer, axis=0)
        self.voxel_position_offset = self.lower_bound
        self.voxel_length = voxel_length


    """"""
    def space_division(self):
        x = math.ceil((self.upper_bound[0] - self.lower_bound[0]) / self.voxel_length) + 1
        y = math.ceil((self.upper_bound[1] - self.lower_bound[1]) / self.voxel_length) + 1
        z = math.ceil((self.upper_bound[2] - self.lower_bound[2]) / self.voxel_length) + 1
        n = x * y * z

        domain_mat = np.zeros((n, 2*4, 4), dtype=np.int32)
        for i in range(n):
            index = i + 1
            # float version
            # domain_mat[i, 0, :] = [index, (i % (x * y) // y) * self.h, (i % (x * y) % y) * self.h,
            #                        i // (x * y) * self.h]
            # int version
            domain_mat[i, 0, :] = [index, (i % (x * y) % x), (i % (x * y) // x), i // (x * y)]
            back = max(0, index - (x * y))
            front = 0 if index + (x * y) > n else index + (x * y)
            pt = i % (x * y)

            # left, right, down, up, left-down, left-up, right-down, right-up
            if pt == 0:
                buf = np.array([0, index + 1, 0, index + x, 0, 0, 0, index + x + 1], dtype=np.int32)
            elif pt == x - 1:
                buf = np.array([index - 1, 0, 0, index + x, 0, index + x - 1, 0, 0], dtype=np.int32)
            elif pt == x * y - x:
                buf = np.array([0, index + 1, index - x, 0, 0, 0, index - x + 1, 0], dtype=np.int32)
            elif pt == x * y - 1:
                buf = np.array([index - 1, 0, index - x, 0, index - x - 1, 0, 0, 0], dtype=np.int32)
            else:
                if pt // x == 0:
                    buf = np.array([index - 1, index + 1, 0, index + x, 0, index + x - 1, 0, index + x + 1],
                                   dtype=np.int32)
                elif pt // x == y - 1:
                    buf = np.array([index - 1, index + 1, index - x, 0, index - x - 1, 0, index - x + 1, 0],
                                   dtype=np.int32)
                elif pt % x == 0:
                    buf = np.array([0, index + 1, index - x, index + x, 0, 0, index - x + 1, index + x + 1],
                                   dtype=np.int32)
                elif pt % x == y - 1:
                    buf = np.array([index - 1, 0, index - x, index + x, index - x - 1, index + x - 1, 0, 0],
                                   dtype=np.int32)
                else:
                    buf = np.array([index - 1, index + 1, index - x, index + x,
                                    index - x - 1, index + x - 1, index - x + 1, index + x + 1], dtype=np.int32)

            contents = np.zeros((2, 8), dtype=np.float32)
            if back != 0:
                contents[0, :] = [pos - x * y if pos != 0 else 0 for pos in buf]
            if front != 0:
                contents[1, :] = [pos + x * y if pos != 0 else 0 for pos in buf]
            contents = contents.T
            contents = contents.reshape(16)
            contents = np.hstack((buf[:4], back, front, buf[4:], contents, 0, 0))
            contents = contents.reshape((7, 4))
            domain_mat[i, 1:8, :] = contents

            # re-arrange
            """
            we set x, y, z have 3 status: -1, 0, 1
            Left = (-1, 0, 0)
            Right = (1, 0, 0)
            Down = (0, -1, 0)
            Up = (0, 1, 0)
            Back = (0, 0, -1)
            Front = (0, 0, 1)
            and other combinations of above, i.e.
            LeftUpBack = (-1, 1, -1) = Left + Up + Back

            re_arrange = [
                ["i", "x", "y", "z"],0-3
                [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)],4-7
                [(0, 0, -1), (0, 0, 1), (-1, -1, 0), (-1, 1, 0)],8-11
                [(1, -1, 0), (1, 1, 0), (-1, 0, -1), (-1, 0, 1)],12-15
                [(1, 0, -1), (1, 0, 1), (0, -1, -1), (0, -1, 1)],16-19
                [(0, 1, -1), (0, 1, 1), (-1, -1, -1), (-1, -1, 1)],20-23
                [(-1, 1, -1), (-1, 1, 1), (1, -1, -1)", (1, -1, 1)],24-27
                [(1, 1, -1), (1, 1, 1), "0", "0"],28-31
            ]

            """
            re_arrange = [
                ["i", "x", "y", "z"],
                ["Left", "Right", "Down", "Up"],
                ["Back", "Front", "LeftDown", "LeftUp"],
                ["RightDown", "RightUp", "LeftBack", "LeftFront"],
                ["RightBack", "RightFront", "DownBack", "DownFront"],
                ["UpBack", "UpFront", "LeftDownBack", "LeftDownFront"],
                ["LeftUpBack", "LeftUpFront", "RightDownBack", "RightDownFront"],
                ["RightUpBack", "RightUpFront", "0", "0"],
            ]






        output_buffer = np.vstack((voxel_matrices for voxel_matrices in domain_mat))
        return output_buffer

    def create_voxels(self):
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        x = int(np.ceil((upper_bound[0] - lower_bound[0]) / self.voxel_length) + 1)
        y = int(np.ceil((upper_bound[1] - lower_bound[1]) / self.voxel_length) + 1)
        z = int(np.ceil((upper_bound[2] - lower_bound[2]) / self.voxel_length) + 1)
        n = x * y * z
        yz = y * z

        voxels = np.zeros([n, 2 * 4, 4], dtype=np.int32)

        for x_id in range(x):
            for y_id in range(y):
                for z_id in range(z):
                    voxel_id = x_id * yz + y_id * z + z_id
                    voxels[voxel_id][0, :] = np.array([voxel_id + 1, x_id, y_id, z_id], dtype=np.int32)
                    left = (x_id - 1) * yz + y_id * z + z_id + 1
                    right = (x_id + 1) * yz + y_id * z + z_id + 1
                    down = x_id * yz + (y_id - 1) * z + z_id + 1
                    up = x_id * yz + (y_id + 1) * z + z_id + 1
                    back = x_id * yz + y_id * z + (z_id - 1) + 1
                    front = x_id * yz + y_id * z + (z_id + 1) + 1
                    left_down = (x_id - 1) * yz + (y_id - 1) * z + z_id + 1
                    left_up = (x_id - 1) * yz + (y_id + 1) * z + z_id + 1
                    right_down = (x_id + 1) * yz + (y_id - 1) * z + z_id + 1
                    right_up = (x_id + 1) * yz + (y_id + 1) * z + z_id + 1
                    left_back = (x_id - 1) * yz + y_id * z + (z_id - 1) + 1
                    left_front = (x_id - 1) * yz + y_id * z + (z_id + 1) + 1
                    right_back = (x_id + 1) * yz + y_id * z + (z_id - 1) + 1
                    right_front = (x_id + 1) * yz + y_id * z + (z_id + 1) + 1
                    down_back = x_id * yz + (y_id - 1) * z + (z_id - 1) + 1
                    down_front = x_id * yz + (y_id - 1) * z + (z_id + 1) + 1
                    up_back = x_id * yz + (y_id + 1) * z + (z_id - 1) + 1
                    up_front = x_id * yz + (y_id + 1) * z + (z_id + 1) + 1
                    left_down_back = (x_id - 1) * yz + (y_id - 1) * z + (z_id - 1) + 1
                    left_down_front = (x_id - 1) * yz + (y_id - 1) * z + (z_id + 1) + 1
                    left_up_back = (x_id - 1) * yz + (y_id + 1) * z + (z_id - 1) + 1
                    left_up_front = (x_id - 1) * yz + (y_id + 1) * z + (z_id + 1) + 1
                    right_down_back = (x_id + 1) * yz + (y_id - 1) * z + (z_id - 1) + 1
                    right_down_front = (x_id + 1) * yz + (y_id - 1) * z + (z_id + 1) + 1
                    right_up_back = (x_id + 1) * yz + (y_id + 1) * z + (z_id - 1) + 1
                    right_up_front = (x_id + 1) * yz + (y_id + 1) * z + (z_id + 1) + 1
                    if x_id == 0:
                        left, left_down, left_up, left_back, left_front, left_down_back, left_down_front, left_up_back, left_up_front = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    if x_id == x - 1:
                        right, right_down, right_up, right_back, right_front, right_down_back, right_down_front, right_up_back, right_up_front = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    if y_id == 0:
                        down, down_back, down_front, left_down, right_down, left_down_back, left_down_front, right_down_back, right_down_front = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    if y_id == y - 1:
                        up, up_back, up_front, left_up, right_up, left_up_back, left_up_front, right_up_back, right_up_front = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    if z_id == 0:
                        back, left_back, right_back, down_back, up_back, left_down_back, left_up_back, right_down_back, right_up_back = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    if z_id == z - 1:
                        front, left_front, right_front, down_front, up_front, left_down_front, left_up_front, right_down_front, right_up_front = 0, 0, 0, 0, 0, 0, 0, 0, 0

                    voxels[voxel_id][1, :] = np.array([left, right, down, up], dtype=np.int32)
                    voxels[voxel_id][2, :] = np.array([back, front, left_down, left_up], dtype=np.int32)
                    voxels[voxel_id][3, :] = np.array([right_down, right_up, left_back, left_front], dtype=np.int32)
                    voxels[voxel_id][4, :] = np.array([right_back, right_front, down_back, down_front], dtype=np.int32)
                    voxels[voxel_id][5, :] = np.array([up_back, up_front, left_down_back, left_down_front],
                                                      dtype=np.int32)
                    voxels[voxel_id][6, :] = np.array([left_up_back, left_up_front, right_down_back, right_down_front],
                                                      dtype=np.int32)
                    voxels[voxel_id][7, :] = np.array([right_up_back, right_up_front, 0, 0], dtype=np.int32)

        return np.vstack((voxel_matrices for voxel_matrices in voxels))


if __name__ == "__main__":
    V = Voxelization(np.array([[0,0,0], [3,3,3]], dtype=np.float32), 0.15)
    import time
    t1 = time.time()
    buf1 = V.space_division()
    print(time.time()-t1)
    t2 = time.time()
    buf2 = V.create_voxels()
    print(time.time() - t2)
