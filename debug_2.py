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
