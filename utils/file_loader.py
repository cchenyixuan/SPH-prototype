import sys
from typing import Tuple
import numpy as np


def load_obj(file):
    print(f"Loading: {file}.")
    vertices = []
    faces = []
    faces_uvs = []
    faces_normals = []
    uvs = []
    normals = []
    vertices_enhanced = []
    try:
        with open(file, "r", encoding="utf-8") as f:
            for row in f:
                if row[:2] == "v ":
                    vertices.append(row[2:-1].split(" "))
                    vertices_enhanced.append(row[2:-1].split(" "))
                elif row[:3] == "vt ":
                    uvs.append(row[3:-1].split(" "))
                elif row[:3] == "vn ":
                    normals.append(row[3:-1].split(" "))
                elif row[:2] == "f ":
                    faces.append([item.split("/")[0] for item in row[2:-1].split(" ")])
                    faces_uvs.append([item.split("/")[1] if item.split("/")[1] else 1 for item in row[2:-1].split(" ")])
                    faces_normals.append([item.split("/")[2] for item in row[2:-1].split(" ")])
            f.close()
    except FileNotFoundError:
        print("File Not Found!", file=sys.stderr)
        pass
    print(f"Loading '{file}' finished with {len(vertices)} vertices and {len(faces)} facets.")
    if not uvs:
        uvs.append([0.0, 0.0])

    vertices = np.array(vertices, dtype=np.float32)  # vertices
    faces = np.array(faces, dtype=np.int32) - np.array([1, 1, 1], dtype=np.int32)  # facets
    normals = np.array(normals, dtype=np.float32)  # normals
    uvs = np.array(uvs, dtype=np.float32)  # uvs


    faces_uvs = np.array(faces_uvs, dtype=np.int32) - np.array([1, 1, 1], dtype=np.int32)
    faces_normals = np.array(faces_normals, dtype=np.int32) - np.array([1, 1, 1], dtype=np.int32)

    for facet, facet_uv, facet_normal in zip(faces, faces_uvs, faces_normals):
        for vertex_index, uv_index, normal_index in zip(facet, facet_uv, facet_normal):
            vertices_enhanced[vertex_index].extend(uvs[uv_index])  # TODO
            vertices_enhanced[vertex_index].extend(normals[normal_index])
    vertices_enhanced = [np.array(item, dtype=np.float32) for item in vertices_enhanced]
    vertices_enhanced_averaged = []
    for vertex in vertices_enhanced:
        pos = vertex[:3]
        uvnnn = np.mean(vertex[3:].reshape((-1, 5)), axis=0)
        vertices_enhanced_averaged.append([*pos, *uvnnn])
    vertices_enhanced_averaged = np.array(vertices_enhanced_averaged, dtype=np.float32)
    return vertices, faces, normals, uvs, vertices_enhanced_averaged


if __name__ == "__main__":
    a, b, c, d, e = load_obj(r"D:\ProgramFiles\PycharmProject\SPH-prototype\models\delta-sph-test/boundary2.obj")