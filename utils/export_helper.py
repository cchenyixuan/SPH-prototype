import numpy as np
import csv
import re
import os
from multiprocessing import Pool


def load_file(file):
    find_vertex = re.compile(r"v (\+?-?[\d.]+) (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
    find_normal = re.compile(r"vn (\+?-?[\d.]+) (\+?-?[\d.]+) (\+?-?[\d.]+)\n", re.S)
    find_facet = re.compile(r"f ([\d.]+)/([\d.]?)/([\d.]+) ([\d.]+)/([\d.]?)/([\d.]+) ([\d.]+)/([\d.]?)/([\d.]+)\n", re.S)
    v_data = []
    n_data = []
    f_data = []
    with open(file, "r", encoding="utf-8") as f:
        for row in f:
            if row[:2] == "v ":
                ans_v = re.findall(find_vertex, row)
                ans_v = [float(ans_v[0][i]) for i in range(3)]
                v_data.append([ans_v[0], ans_v[1], ans_v[2]])
            if row[:2] == "vn":
                ans_n = re.findall(find_normal, row)
                ans_n = [float(ans_n[0][i]) for i in range(3)]
                n_data.append([ans_n[0], ans_n[1], ans_n[2]])
            if row[:2] == "f ":
                ans_f = re.findall(find_facet, row)
                ans_f = [[int(ans_f[0][i*3+j])-1 for j in [0, 2]] for i in range(3)]
                f_data.append([ans_f[0], ans_f[1], ans_f[2]])
        f.close()
    return np.array(v_data, dtype=np.float32), np.array(n_data, dtype=np.float32), np.array(f_data, dtype=np.uint32)


class ExportConverter:
    def __init__(self, npy_file_name, obj_file_name, case_name="", output_dir=".", scale=1.0, offset=np.zeros((3,), dtype=np.float32)):
        self.data = np.load(npy_file_name).reshape((-1, 4, 4))
        self.node_number = self.data.shape[0]
        self.position = np.array([self.data[i, 0, :3] for i in range(self.node_number)], dtype=np.float32)
        self.raw_sph_force = np.array([self.data[i, 3, :3] for i in range(self.node_number)], dtype=np.float32)
        self.pressure = np.array([self.data[i, 3, 3] for i in range(self.node_number)], dtype=np.float32)
        self.obj_data_position, self.obj_data_normal, self.obj_data_facet = load_file(obj_file_name)
        # self.obj_data_position is identical with self.position, thus we can get its normal
        # ------sanity check-----------
        self._check_file_sanity()
        # ------sanity check finish----
        self.normal = self._get_normal()
        # ------compute sph-force------
        self.non_alpha_sph_force, self.sph_force = self._calculate_sph_force()
        # ------transform to original position-----
        self.position *= scale
        self.position += offset
        # ------export-----
        self.output_dir = output_dir
        self.case_name = case_name
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, i):
        # csv file
        with open(rf"{self.output_dir}/{self.case_name}_output_with_normal_{i}.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["x", "y", "z", "p", "nx", "ny", "nz", "fx", "fy", "fz", "e1", "e2"])
            csv_writer.writerow(["T = 0 Boundary_1 WALL"])
            for vertex, pressure, sph_force, normal in zip(self.position, self.pressure, self.sph_force, self.normal):
                csv_writer.writerow([*vertex, pressure, *normal, *sph_force, 0.0, 0.0])
            f.close()

    def export_stl(self, i):
        # stl file
        """ facet normal 0.121312 -0.177241 -0.976662
            outer loop
            vertex -8.108192 4.192158 -6.288399
            vertex -8.121511 4.178309 -6.287540
            vertex -8.121511 4.191253 -6.289889
            endloop
            endfacet"""
        with open(rf"{self.output_dir}/{self.case_name}_output_with_normal_{i}.stl", "w", newline="") as f:
            for vertex, pressure, sph_force, normal in zip(self.position, self.pressure, self.sph_force, self.normal):
                f.write(f"facet normal {sph_force[0]} {sph_force[1]} {sph_force[2]}\n")
                f.write(f"outer loop\n")
                f.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                f.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                f.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                f.write(f"endloop\n")
                f.write(f"endfacet\n")
            f.close()

    def _check_file_sanity(self):
        try:
            assert self.obj_data_position.shape[0] == self.node_number
            assert np.sum(self.position-self.obj_data_position) < 1.0
        except AssertionError:
            print("Sanity check failed!")
        finally:
            print("Sanity check finished!")

    def _get_normal(self):
        normal = np.zeros((self.node_number, 4), dtype=np.float32)
        for facet in self.obj_data_facet:
            for node in facet:
                position_id, normal_id = node
                normal[position_id][:3] += self.obj_data_normal[normal_id]
                normal[position_id][3] += 1
        for step in range(normal.shape[0]):
            if normal[step, 3] != 0.0:
                normal[step, :3] /= normal[step, 3]
        return normal[:, :3]

    def _calculate_sph_force(self):
        def get_alpha(norm):
            if norm < 1:
                return norm
            else:
                return 1.0
        # normalize sph_force
        sph_force = [[np.linalg.norm(item), *item] for step, item in enumerate(self.raw_sph_force)]
        raw_data = np.array(sph_force, dtype=np.float32)
        sph_force.sort()
        threshold = sph_force[int(0.9*self.node_number)][0]
        for step, item in enumerate(raw_data):
            if item[0] > threshold:
                raw_data[step][0] = threshold

        raw_data /= np.mean(raw_data[:, 0])
        raw_data *= 4
        print(np.mean(raw_data[:, 0]), np.max(raw_data[:, 0]), np.min(raw_data[:, 0]))
        ans = []
        for i in range(self.node_number):
            alpha = get_alpha(raw_data[i][0])
            ans.append((max(1-alpha, 0))*self.normal[i]+alpha*raw_data[i][1:])
        return raw_data[:, 1:], np.array(ans, dtype=np.float32)


if __name__ == "__main__":
    export = ExportConverter(r"C:\Users\cchen\PycharmProjects\SPH-prototype-multi-version/0.4898775.npy", r"H03_boundary.obj", case_name="H03", output_dir=r"./export_stl", scale=100.0, offset=np.array([40.8099, -20.3448, -244.12], dtype=np.float32))
    pool = Pool(34)
    pool.map(export.export, [i for i in range(2)])
    pool.close()
    print("Finished")
